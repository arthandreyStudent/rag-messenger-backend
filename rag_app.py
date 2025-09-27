import os
import json
import hashlib
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import re

# Core dependencies
from openai import AzureOpenAI
from dotenv import load_dotenv

# LangChain dependencies
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.schema import Document as LangChainDocument
from langchain.chains import ConversationalRetrievalChain

# Traditional search dependencies
from rank_bm25 import BM25Okapi

# Multilingual dependencies
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator
import spacy

# Set seed for consistent language detection
DetectorFactory.seed = 0

load_dotenv()

@dataclass
class Document:
    """Document structure for RAG system"""
    id: str
    content: str
    metadata: Dict[str, Any]
    created_at: datetime
    language: str = "en"  # Added language field


@dataclass
class ContextualChunk:
    """Contextual chunk with generated context"""
    id: str
    original_content: str
    context: str
    combined_content: str
    embedding: Optional[List[float]]
    doc_id: str
    metadata: Dict[str, Any]
    language: str = "en"  # Added language field
    translated_content: Optional[str] = None  # For multilingual processing


@dataclass
class SearchResult:
    """Search result with relevance scores"""
    chunk: ContextualChunk
    vector_score: float
    bm25_score: float
    rerank_score: float
    final_score: float
    language: str = "en"

# Set seed for consistent language detection
DetectorFactory.seed = 0

class MultilingualHandler:
    """Handles multilingual operations for the RAG system"""

    def __init__(self):
        self.translator = GoogleTranslator()

        # Load spacy model
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("⚠️  English spacy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None

        # Define supported languages
        self.supported_languages = {
            'en': 'English',
            'tl': 'Tagalog',
            'ceb': 'Cebuano'
        }

        # Enhanced language-specific markers for better detection
        self.language_markers = {
            'tl': {
                # Distinctly Tagalog words/phrases
                'exclusive': ['ang', 'mga', 'ng', 'sa', 'na', 'at', 'ay', 'para', 'kung', 'kapag', 'dahil', 'kasi',
                              'pero', 'ngunit', 'ako', 'ikaw', 'siya', 'kami', 'kayo', 'sila', 'ito', 'iyan', 'iyon',
                              'sino', 'ano', 'saan', 'kailan', 'bakit', 'paano', 'inyong', 'ninyo', 'namin', 'natin',
                              'kanila', 'nila', 'niya', 'mo', 'ko', 'tayo', 'hindi', 'wala', 'meron', 'may', 'mayroon',
                              'dapat', 'pwede', 'kailangan', 'gusto', 'ayaw', 'ibig', 'nais', 'salamat', 'pasensya',
                              'opo', 'hindi po', 'oo po'],

                # Tagalog-specific grammatical patterns
                'patterns': [r'\b(ng|sa|na)\s+\w+', r'\b(hindi|wala)\b', r'\b(inyong|ninyo)\b', r'\b(po|opo)\b'],

                # Common Tagalog question words
                'question_words': ['sino', 'ano', 'saan', 'kailan', 'bakit', 'paano', 'ilan', 'magkano'],

                # Tagalog pronouns
                'pronouns': ['ako', 'ikaw', 'siya', 'kami', 'kayo', 'sila', 'tayo', 'namin', 'ninyo', 'nila', 'inyong']
            },

            'ceb': {
                # Distinctly Cebuano words/phrases
                'exclusive': ['ang', 'mga', 'sa', 'og', 'ug', 'kay', 'para', 'kung', 'kon', 'tungod', 'pero', 'apan',
                              'mao', 'ako', 'ikaw', 'siya', 'kami', 'kamo', 'sila', 'kini', 'kana', 'kadto', 'kinsa',
                              'unsa', 'asa', 'kanus-a', 'nganong', 'giunsa', 'inyong', 'ninyo', 'namo', 'nato', 'nila',
                              'niya', 'mo', 'ko', 'kita', 'dili', 'wala', 'naa', 'naay', 'kinahanglan', 'gusto', 'ayaw',
                              'salamat', 'pasayloa', 'oo', 'dili gyud'],

                # Cebuano-specific grammatical patterns
                'patterns': [r'\b(og|ug)\s+\w+', r'\b(dili|wala)\b', r'\b(naa|naay)\b', r'\b(gyud|gud|jud)\b',
                             r'\b(dapit|lugar)\b', r'\b(asa|hain)\b'],

                # Common Cebuano question words
                'question_words': ['kinsa', 'unsa', 'asa', 'kanus-a', 'nganong', 'giunsa', 'pila', 'tagpila', 'hain'],

                # Cebuano pronouns and particles
                'pronouns': ['ako', 'ikaw', 'siya', 'kami', 'kamo', 'sila', 'kita', 'namo', 'ninyo', 'nila'],
                'particles': ['og', 'ug', 'gyud', 'gud', 'jud', 'ra', 'man', 'gani', 'kay']
            }
        }

        # Language-specific stopwords
        self.stopwords = {
            'en': set(
                ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are',
                 'was', 'were']),
            'tl': set(['ang', 'mga', 'sa', 'ng', 'na', 'at', 'ay', 'para', 'kung', 'kapag', 'dahil', 'kasi', 'pero',
                       'ngunit']),
            'ceb': set(['ang', 'mga', 'sa', 'og', 'ug', 'kay', 'para', 'kung', 'kon', 'tungod', 'pero', 'apan', 'mao'])
        }

    def detect_language(self, text: str) -> str:
        """Enhanced language detection with better Tagalog/Cebuano differentiation"""
        if not text or len(text.strip()) < 5:
            return 'en'

        text_clean = text.lower().strip()

        # First, use enhanced keyword-based detection
        detected_lang = self._enhanced_keyword_detection(text_clean)

        # If keyword detection is confident, return it
        if detected_lang != 'unknown':
            print(f"🎯 Enhanced detection: {detected_lang} for '{text[:50]}...'")
            return detected_lang

        # Fallback to langdetect library
        try:
            detected = detect(text)
            if detected in ['tl', 'ceb', 'en']:
                print(f"📚 Langdetect fallback: {detected}")
                return detected
            else:
                # If langdetect gives unexpected result, do pattern analysis
                pattern_result = self._pattern_based_detection(text_clean)
                print(f"🔍 Pattern analysis: {pattern_result}")
                return pattern_result
        except Exception as e:
            print(f"⚠️  Language detection error: {e}")
            return self._pattern_based_detection(text_clean)

    def _enhanced_keyword_detection(self, text: str) -> str:
        """Enhanced keyword-based detection with weighted scoring"""
        scores = {'tl': 0, 'ceb': 0, 'en': 0}
        words = text.split()

        # Check for distinctive markers
        for lang in ['tl', 'ceb']:
            markers = self.language_markers[lang]

            # Score exclusive words (higher weight)
            for word in words:
                if word in markers['exclusive']:
                    scores[lang] += 3

            # Score question words (very distinctive)
            for qword in markers['question_words']:
                if qword in text:
                    scores[lang] += 5

            # Score pronouns
            for pronoun in markers['pronouns']:
                if pronoun in words:
                    scores[lang] += 2

            # Score particles (for Cebuano)
            if 'particles' in markers:
                for particle in markers['particles']:
                    if particle in words:
                        scores[lang] += 4

        # Check for English words
        english_indicators = ['the', 'and', 'or', 'is', 'are', 'was', 'were', 'have', 'has', 'will', 'would', 'what',
                              'where', 'when', 'how', 'why', 'your', 'business', 'hours', 'branch', 'location']
        for word in words:
            if word in english_indicators:
                scores['en'] += 2

        print(f"🔢 Keyword scores: {scores}")

        # Determine winner with minimum threshold
        max_score = max(scores.values())
        if max_score >= 3:  # Minimum confidence threshold
            winner = max(scores, key=scores.get)

            # Additional validation for close scores
            if scores['tl'] > 0 and scores['ceb'] > 0:
                diff = abs(scores['tl'] - scores['ceb'])
                if diff < 2:  # Very close scores, do pattern analysis
                    return self._pattern_based_detection(text)

            return winner

        return 'unknown'

    def _pattern_based_detection(self, text: str) -> str:
        """Pattern-based detection for ambiguous cases"""
        pattern_scores = {'tl': 0, 'ceb': 0}

        # Check specific patterns for each language
        for lang in ['tl', 'ceb']:
            patterns = self.language_markers[lang]['patterns']
            for pattern in patterns:
                matches = re.findall(pattern, text)
                pattern_scores[lang] += len(matches) * 2

        # Specific Cebuano indicators
        cebuano_indicators = [
            r'\b(asa|hain)\b',  # "where" in Cebuano
            r'\b(dapit|lugar)\b',  # "place"
            r'\b(nahimutang|nakit-an)\b',  # "located"
            r'\b(unsa|kinsa)\b',  # "what/who"
            r'\b(gyud|gud|jud)\b',  # Cebuano particles
            r'\b(naa|naay)\b',  # "there is/are"
            r'\b(dili)\b',  # "no/not" in Cebuano
            r'\b(kanus-a)\b'  # "when" in Cebuano
        ]

        # Specific Tagalog indicators
        tagalog_indicators = [
            r'\b(saan|nasaan)\b',  # "where" in Tagalog
            r'\b(inyong|ninyo)\b',  # "your" in Tagalog
            r'\b(hindi)\b',  # "no/not" in Tagalog
            r'\b(kailan)\b',  # "when" in Tagalog
            r'\b(ano|sino)\b',  # "what/who" in Tagalog
            r'\b(meron|mayroon)\b',  # "there is/are"
            r'\b(po|opo)\b'  # Politeness markers in Tagalog
        ]

        for pattern in cebuano_indicators:
            if re.search(pattern, text):
                pattern_scores['ceb'] += 3

        for pattern in tagalog_indicators:
            if re.search(pattern, text):
                pattern_scores['tl'] += 3

        print(f"🎨 Pattern scores: {pattern_scores}")

        if pattern_scores['ceb'] > pattern_scores['tl']:
            return 'ceb'
        elif pattern_scores['tl'] > pattern_scores['ceb']:
            return 'tl'
        else:
            return 'en'  # Default fallback

    def translate_to_english(self, text: str, source_lang: str) -> str:
        """Translate text to English"""
        if source_lang == 'en' or not text.strip():
            return text

        try:
            # Map language codes for translator
            lang_mapping = {'tl': 'tl', 'ceb': 'ceb', 'en': 'en'}
            source_code = lang_mapping.get(source_lang, source_lang)

            translated = self.translator.translate(text, source=source_code, target='en')
            return translated if translated else text

        except Exception as e:
            print(f"Translation error (to English): {e}")
            return text

    def translate_from_english(self, text: str, target_lang: str) -> str:
        """Translate text from English to target language"""
        if target_lang == 'en' or not text.strip():
            return text

        try:
            # Map language codes for translator
            lang_mapping = {'tl': 'tl', 'ceb': 'ceb', 'en': 'en'}
            target_code = lang_mapping.get(target_lang, target_lang)

            translated = self.translator.translate(text, source='en', target=target_code)
            return translated if translated else text

        except Exception as e:
            print(f"Translation error (from English): {e}")
            return text

    def get_stopwords(self, language: str) -> set:
        """Get stopwords for specific language"""
        return self.stopwords.get(language, self.stopwords['en'])

    def preprocess_text_multilingual(self, text: str, language: str) -> list:
        """Preprocess text for multilingual search"""
        if not text:
            return []

        # Basic tokenization and cleaning
        text = text.lower().strip()

        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()

        # Remove stopwords
        stopwords = self.get_stopwords(language)
        tokens = [token for token in tokens if token not in stopwords and len(token) > 1]

        return tokens

    def get_fallback_response(self, language: str) -> str:
        """Get fallback response in appropriate language"""
        responses = {
            'en': "I apologize, but I'm experiencing technical difficulties. Please try again later.",
            'tl': "Pasensya po, ngunit nagkakaroon ako ng teknikal na problema. Subukan na lang po ulit mamaya.",
            'ceb': "Pasensya kaayo, nagka-teknikal nga problema kami karon. Palihug pagsulay pag-usab unya."
        }
        return responses.get(language, responses['en'])


class PromptCache:
    """Simple in-memory prompt caching system"""

    def __init__(self, ttl_hours: int = 24):
        self.cache = {}
        self.ttl_hours = ttl_hours

    def get(self, prompt: str) -> Optional[str]:
        key = hashlib.md5(prompt.encode()).hexdigest()
        if key in self.cache:
            response, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(hours=self.ttl_hours):
                return response
            else:
                del self.cache[key]
        return None

    def set(self, prompt: str, response: str):
        key = hashlib.md5(prompt.encode()).hexdigest()
        self.cache[key] = (response, datetime.now())

    def clear_expired(self):
        current_time = datetime.now()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if current_time - timestamp >= timedelta(hours=self.ttl_hours)
        ]
        for key in expired_keys:
            del self.cache[key]


class ChromaVectorDB:
    """Multilingual Chroma vector database class using LangChain integration"""

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory

        # Initialize Azure OpenAI embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            model=os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_version="2024-12-01-preview",
            chunk_size=32
        )

        self.vectorstore = None
        self.chunks = []

        # Ensure the persist directory exists
        os.makedirs(persist_directory, exist_ok=True)

    def add_chunks(self, chunks: List[ContextualChunk]):
        """Add multilingual contextual chunks to Chroma vector database"""
        if not chunks:
            return

        self.chunks = chunks

        # Prepare texts and metadatas for Chroma (use English for embeddings)
        texts = []
        metadata = []

        for chunk in chunks:
            # Use translated content for embeddings if available, otherwise combined content
            embedding_text = chunk.translated_content if chunk.translated_content else chunk.combined_content
            texts.append(embedding_text)

            metadata.append({
                "chunk_id": chunk.id,
                "doc_id": chunk.doc_id,
                "original_content": chunk.original_content[:500],
                "context": chunk.context,
                "language": chunk.language,
                "translated_content": chunk.translated_content[:500] if chunk.translated_content else "",
                **chunk.metadata
            })

        # Create Chroma vectorstore
        self.vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadata,
            persist_directory=self.persist_directory
        )

    def search(self, query: str, k: int = 10, query_language: str = 'en') -> List[Tuple[ContextualChunk, float]]:
        """Search for similar vectors using semantic similarity"""
        if not self.vectorstore or not self.chunks:
            return []

        try:
            # Perform similarity search with scores (query should be in English for embeddings)
            results = self.vectorstore.similarity_search_with_score(query, k=k)

            # Convert results back to ContextualChunk format
            search_results = []
            for doc, score in results:
                # Find corresponding chunk by metadata
                chunk_id = doc.metadata.get("chunk_id", "")
                matching_chunk = next((chunk for chunk in self.chunks if chunk.id == chunk_id), None)

                if matching_chunk:
                    search_results.append((matching_chunk, float(score)))

            return search_results

        except Exception as e:
            print(f"Error in vector search: {e}")
            return []

    def load_existing(self):
        """Load existing Chroma database"""
        try:
            if os.path.exists(self.persist_directory):
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                return True
        except Exception as e:
            print(f"Error loading existing database: {e}")
        return False


class MultilingualBM25:
    """Multilingual BM25 implementation for contextual chunks"""

    def __init__(self, multilingual_handler: MultilingualHandler):
        self.bm25_indices = {}  # Separate BM25 for each language
        self.chunks_by_language = {}  # Chunks organized by language
        self.multilingual_handler = multilingual_handler

    def fit(self, chunks: List[ContextualChunk]):
        """Fit BM25 on contextual chunks, organized by language"""
        # Organize chunks by language
        self.chunks_by_language = {'en': [], 'tl': [], 'ceb': []}

        for chunk in chunks:
            lang = chunk.language
            if lang not in self.chunks_by_language:
                self.chunks_by_language[lang] = []
            self.chunks_by_language[lang].append(chunk)

        # Create separate BM25 index for each language
        for lang, lang_chunks in self.chunks_by_language.items():
            if lang_chunks:
                corpus = []
                for chunk in lang_chunks:
                    # Use appropriate content based on language
                    if lang == 'en':
                        content = chunk.translated_content if chunk.translated_content else chunk.combined_content
                    else:
                        content = chunk.combined_content

                    processed_text = self.multilingual_handler.preprocess_text_multilingual(content, lang)
                    corpus.append(processed_text)

                if corpus:
                    self.bm25_indices[lang] = BM25Okapi(corpus)

    def search(self, query: str, query_language: str, k: int = 10) -> List[Tuple[ContextualChunk, float]]:
        """Search using multilingual BM25"""
        results = []

        # Search in the same language first
        if query_language in self.bm25_indices and query_language in self.chunks_by_language:
            lang_results = self._search_language(query, query_language, k)
            results.extend(lang_results)

        # If not enough results, search in English as fallback
        if len(results) < k and query_language != 'en' and 'en' in self.bm25_indices:
            # Translate query to English for cross-language search
            english_query = self.multilingual_handler.translate_to_english(query, query_language)
            english_results = self._search_language(english_query, 'en', k - len(results))
            results.extend(english_results)

        # Sort by score and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def _search_language(self, query: str, language: str, k: int) -> List[Tuple[ContextualChunk, float]]:
        """Search within a specific language"""
        if language not in self.bm25_indices or language not in self.chunks_by_language:
            return []

        bm25 = self.bm25_indices[language]
        chunks = self.chunks_by_language[language]

        query_tokens = self.multilingual_handler.preprocess_text_multilingual(query, language)
        if not query_tokens:
            return []

        scores = bm25.get_scores(query_tokens)

        # Get top k results
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only return results with positive scores
                results.append((chunks[idx], float(scores[idx])))

        return results


class MultilingualRAGChatbot:
    """Main multilingual RAG-based AI Chatbot class"""

    def __init__(self,
                 azure_api_key: str = None,
                 azure_endpoint: str = None,
                 api_version: str = "2024-12-01-preview",
                 deployment_name: str = None,
                 embedding_deployment: str = None):

        # Initialize multilingual handler
        self.multilingual_handler = MultilingualHandler()

        # Azure configuration
        self.azure_api_key = azure_api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = api_version
        self.deployment_name = deployment_name or os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-35-turbo")
        self.embedding_deployment = embedding_deployment or os.getenv("AZURE_EMBEDDING_DEPLOYMENT",
                                                                      "text-embedding-ada-002")

        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
            api_key=self.azure_api_key
        )

        # Initialize multilingual components
        self.vector_db = ChromaVectorDB()
        self.bm25 = MultilingualBM25(self.multilingual_handler)
        self.prompt_cache = PromptCache()

        # Initialize LangChain Azure components
        self.embeddings = AzureOpenAIEmbeddings(
            model=self.embedding_deployment,
            azure_endpoint=self.azure_endpoint,
            api_key=self.azure_api_key,
            openai_api_version=self.api_version,
            chunk_size=32
        )

        self.llm = AzureChatOpenAI(
            deployment_name=self.deployment_name,
            openai_api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
            openai_api_key=self.azure_api_key,
            temperature=0.1
        )

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            keep_separator=True,
            length_function=len,
            is_separator_regex=False
        )

        # Model configurations
        self.embedding_model = "text-embedding-ada-002"
        self.llm_model = "gpt-35-turbo"

        # Multilingual system prompts
        self.system_prompts = {
            'en': """You are a helpful customer service assistant for a small-medium enterprise (SME) business.
You handle customer inquiries via Facebook Messenger with a friendly, professional tone.

IMPORTANT: Always respond in English only.

Guidelines:
- Keep responses to 1-2 sentences maximum.
- Be direct and concise while remaining helpful
- Use the provided context to answer questions accurately
- If you cannot find relevant information in the context, say so politely
- Maintain a conversational, approachable tone
- Focus on solving customer problems quickly with brief answers""",

            'tl': """Ikaw ay isang matulunging customer service assistant para sa isang small-medium enterprise (SME) business.
Ikaw ay sumasagot sa mga tanong ng customer sa pamamagitan ng Facebook Messenger nang magiliw at propesyonal.

MAHALAGANG PAALALA: Sumagot ka LAGI sa wikang FIlipino/Tagalog lamang.

Mga gabay:
- Panatilihin ang mga sagot na 1-2 pangungusap lamang.
- Maging direkta at malinaw habang nagiging matulungin
- Gamitin ang nakalagay na context upang sumagot nang tama
- Kung hindi mo mahanap ang kaugnay na impormasyon sa context, sabihin ito nang magalang
- Panatilihin ang makakausap at maaasahang tono
- Tumuon sa paglutas ng mga problema ng customer nang mabilis gamit ang maikling sagot""",

            'ceb': """Ikaw usa ka matinabangon nga customer service assistant para sa usa ka small-medium enterprise (SME) nga negosyo or kompanya.
Ikaw ang mutubag sa mga pangutana sa customer pinaagi sa Facebook Messenger nga mabination ug propesyonal nga tono.

IMPORTANTE NA PAHINUMDOM: Dapat lang ka motubag sa pinulongang Cebuano/Binisaya KANUNAY.
 
Mga giya:
- Limitaha ang mga tubag sa 1-2 ka hugot nga mga linya ang tubag
- Pagpaka-direkta ug klaro sa imong mga tubag samtang matinabangon gihapon ikaw
- Gamita ang gihatag nga konteksto aron matubag ang mga pangutana sa hustong paagi
- Kung dili nimo makit-an ang may kalabotan nga kasayuran sa kontektso, sultiha kini nga matinahuron
- Pabilin nga makig-istorya ug maasahan nga tono
- Pagpokus sa dali nga pagsulbad sa problema sa customer pinaagi sa mubo nga tubag"""
        }

        # Initialize knowledge base
        self.is_initialized = False
        self.initialize_knowledge_base()

    def detect_document_language(self, content: str) -> str:
        """Detect the primary language of a document"""
        return self.multilingual_handler.detect_language(content)

    def generate_context_for_chunk_multilingual(self, chunk: str, document: str, language: str) -> str:
        """Generate contextual information for a chunk in multiple languages"""

        # Translate to English for LLM processing if needed
        if language != 'en':
            english_chunk = self.multilingual_handler.translate_to_english(chunk, language)
            english_document = self.multilingual_handler.translate_to_english(document[:2000], language)
        else:
            english_chunk = chunk
            english_document = document[:2000]

        prompt = f"""Please provide a brief, informative context for the following text chunk from a customer service document.

The context should explain what this chunk is about and how it relates to the overall document in 1-2 sentences.

Full document (excerpt): {english_document}...

Text chunk: {english_chunk}

Context:"""

        # Check cache first
        cached_response = self.prompt_cache.get(prompt)
        if cached_response:
            context = cached_response
        else:
            try:
                from langchain.schema import HumanMessage, SystemMessage

                messages = [
                    SystemMessage(
                        content="You are an expert at creating concise, informative contexts for text chunks."),
                    HumanMessage(content=prompt)
                ]

                response = self.llm.invoke(messages)
                context = response.content.strip()
                self.prompt_cache.set(prompt, context)

            except Exception as e:
                print(f"Error generating context: {e}")
                context = "This chunk contains customer service information."

        # Translate context back to original language if needed
        if language != 'en':
            context = self.multilingual_handler.translate_from_english(context, language)

        return context

    def create_multilingual_contextual_chunks(self, documents: List[Document]) -> List[ContextualChunk]:
        """Create multilingual contextual chunks from documents"""
        contextual_chunks = []

        for doc in documents:
            print(f"Processing document: {doc.id} (Language: {doc.language})")

            # Convert to LangChain document format
            langchain_doc = LangChainDocument(
                page_content=doc.content,
                metadata=doc.metadata
            )

            # Split document
            chunks = self.text_splitter.split_documents([langchain_doc])
            print(f"Split document {doc.id} into {len(chunks)} chunks")

            # Generate contextual information for each chunk
            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i + 1}/{len(chunks)} for document {doc.id}")

                # Generate context in the document's language
                context = self.generate_context_for_chunk_multilingual(
                    chunk.page_content, doc.content, doc.language
                )

                # Combine context with original chunk (in original language)
                combined_content = f"Context: {context}\n\nContent: {chunk.page_content}"

                # For multilingual embedding, translate to English
                if doc.language != 'en':
                    translated_combined = self.multilingual_handler.translate_to_english(
                        combined_content, doc.language
                    )
                else:
                    translated_combined = combined_content

                # Get embedding for English version
                try:
                    embedding = self.embeddings.embed_query(translated_combined)
                except Exception as e:
                    print(f"Error getting embedding for chunk: {e}")
                    embedding = []

                # Create multilingual contextual chunk
                contextual_chunk = ContextualChunk(
                    id=f"{doc.id}_chunk_{i}",
                    original_content=chunk.page_content,
                    context=context,
                    combined_content=combined_content,
                    embedding=embedding,
                    doc_id=doc.id,
                    language=doc.language,
                    translated_content=translated_combined if doc.language != 'en' else None,
                    metadata={
                        **doc.metadata,
                        **chunk.metadata,
                        'chunk_index': i,
                        'chunk_size': len(chunk.page_content),
                        'language': doc.language,
                        'has_translation': doc.language != 'en'
                    }
                )

                contextual_chunks.append(contextual_chunk)

        return contextual_chunks

    def build_knowledge_base(self, documents: List[Document]):
        """Build multilingual knowledge base from documents"""
        print("Creating multilingual contextual chunks...")
        contextual_chunks = self.create_multilingual_contextual_chunks(documents)

        print("Building vector database...")
        self.vector_db.add_chunks(contextual_chunks)

        print("Building multilingual BM25 index...")
        self.bm25.fit(contextual_chunks)

        print(f"Multilingual knowledge base built with {len(contextual_chunks)} contextual chunks")

    def multilingual_search(self, query: str, user_language: str, k: int = 5) -> List[SearchResult]:
        """Hybrid multilingual search combining vector similarity and BM25"""

        # Translate query to English for vector search
        english_query = self.multilingual_handler.translate_to_english(query, user_language)

        # Vector search using English query
        vector_results = self.vector_db.search(english_query, k=20, query_language=user_language)

        # Multilingual BM25 search
        bm25_results = self.bm25.search(query, user_language, k=20)

        # Combine results
        combined_results = {}

        # Add vector results
        for chunk, score in vector_results:
            combined_results[chunk.id] = SearchResult(
                chunk=chunk,
                vector_score=score,
                bm25_score=0.0,
                rerank_score=0.0,
                final_score=score,
                language=chunk.language
            )

        # Add BM25 results
        for chunk, score in bm25_results:
            if chunk.id in combined_results:
                combined_results[chunk.id].bm25_score = score
            else:
                combined_results[chunk.id] = SearchResult(
                    chunk=chunk,
                    vector_score=0.0,
                    bm25_score=score,
                    rerank_score=0.0,
                    final_score=score,
                    language=chunk.language
                )

        # Calculate combined scores with language preference
        results = list(combined_results.values())
        for result in results:
            # Give preference to same-language results
            language_bonus = 0.2 if result.chunk.language == user_language else 0.0
            result.final_score = (0.6 * result.vector_score + 0.4 * result.bm25_score + language_bonus)

        # Sort by combined score
        results.sort(key=lambda x: x.final_score, reverse=True)

        # Rerank top results
        reranked_results = self.multilingual_rerank(query, user_language, results, top_k=k)

        return reranked_results

    def multilingual_rerank(self, query: str, user_language: str, results: List[SearchResult], top_k: int = 5) -> List[
        SearchResult]:
        """Rerank search results using multilingual LLM"""
        if len(results) <= top_k:
            return sorted(results, key=lambda x: x.final_score, reverse=True)

        # Translate query to English for reranking
        english_query = self.multilingual_handler.translate_to_english(query, user_language)

        # Prepare reranking prompt with multilingual content
        candidates = []
        for i, result in enumerate(results[:20]):
            # Use appropriate content for reranking
            if result.chunk.language == 'en' or result.chunk.translated_content:
                content = result.chunk.translated_content or result.chunk.combined_content
            else:
                content = result.chunk.combined_content

            candidates.append(f"[{i}] {content[:500]}...")

        rerank_prompt = f"""Given the user query and the following text candidates, rank them by relevance to the query.
Return only the indices of the top {top_k} most relevant candidates, in order of relevance.

Query: {english_query}

Candidates:
{chr(10).join(candidates)}

Top {top_k} most relevant indices (comma-separated):"""

        try:
            from openai.types.chat import (
                ChatCompletionSystemMessageParam,
                ChatCompletionUserMessageParam
            )

            messages: List[ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam] = [
                {"role": "system",
                 "content": "You are an expert at ranking text relevance for search queries across multiple languages."},
                {"role": "user", "content": rerank_prompt}
            ]

            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.1,
                max_tokens=200
            )

            # Parse reranking results
            rerank_indices_str = response.choices[0].message.content.strip()
            rerank_indices = [int(x.strip()) for x in rerank_indices_str.split(',') if x.strip().isdigit()]

            # Apply reranking scores
            reranked_results = []
            for rank, idx in enumerate(rerank_indices[:top_k]):
                if idx < len(results):
                    result = results[idx]
                    result.rerank_score = (top_k - rank) / top_k
                    result.final_score = (result.vector_score + result.bm25_score + result.rerank_score) / 3
                    reranked_results.append(result)

            return reranked_results

        except Exception as e:
            print(f"Error in multilingual reranking: {e}")
            return sorted(results[:top_k], key=lambda x: x.final_score, reverse=True)

    def generate_response(self, query: str, max_tokens: int = 200, user_language: str = None) -> Dict[str, Any]:
        """Generate multilingual response to user query"""

        # Use provided user_language or detect from query
        if not user_language:
            user_language = self.multilingual_handler.detect_language(query)
        print(f"🌍 Detected language: {user_language}")

        # Search for relevant context
        search_results = self.multilingual_search(query, user_language, k=5)

        if not search_results:
            fallback_response = self.multilingual_handler.get_fallback_response(user_language)
            return {
                "response": fallback_response,
                "confidence": 0.0,
                "sources": [],
                "language": user_language
            }

        # Prepare context from search results
        context_parts = []
        sources = []

        for result in search_results:
            content = result.chunk.original_content
            context_parts.append(f"Source: {content}")
            sources.append({
                "content": content[:200] + "..." if len(content) > 200 else content,
                "score": result.final_score,
                "language": result.chunk.language
            })

        context = "\n\n---\n\n".join(context_parts)

        # Translate query to English for consistency with context
        english_query = self.multilingual_handler.translate_to_english(query, user_language)

        # Get the appropriate system prompt for the detected language
        system_prompt = self.system_prompts.get(user_language, self.system_prompts['en'])

        # Generate language-specific response prompts
        if user_language == 'en':
            response_prompt = f"""Context information:
    {context}

    User question: {query}

    Please provide a helpful, accurate response in 1-2 sentences maximum based on the context above.
     Do NOT repeat or summarize the context. Only answer the user's question directly.
     If the context doesn't contain sufficient information to answer the question, please say so politely."""

        elif user_language == 'tl':
            response_prompt = f"""Answer the user's question in Filipino/Tagalog using the provided context.

    Context (in English):
    {context}
    
    User question (translated to English): {english_query}
    Original question (in Filipino): {query}

    Magbigay ng matulungin at tumpak na sagot na 1-2 pangungusap lamang base sa konteksto sa itaas.
     Huwag ulitin o ibuod ang konteksto. Sagutin lamang ang tanong ng user nang direkta.
     Kung walang sapat na impormasyon sa konteksto, sabihin ito nang magalang sa wikang Filipino/Tagalog."""

        elif user_language == 'ceb':
            response_prompt = f"""Answer the user's question in Cebuano/Binisaya using the provided context.

    Context (in English):
    {context}
    
    User question (translated to English): {english_query}
    Original question (in Cebuano): {query}

    Hatagi ug matinabangon ug tukma nga tubag nga 1-2 ka linya lang base sa konteksto sa ibabaw.
     Ayaw isulti o i-summarize ang konteksto sa imohang tubag. Tubaga lang og diretso ang pangutana sa user.
     Kung walay igo nga kasayuran sa konteksto, sultihi kini nga matinahuron sa pinulongang Cebuano/Binisaya."""

        else:
            # Fallback for any other language
            response_prompt = f"""Context information:
    {context}

    User question: {query}

    Please provide a helpful, accurate response in 1-2 sentences maximum based on the context above.
     Do NOT repeat or summarize the context. Only answer the user's question directly.
     If the context doesn't contain sufficient information to answer the question, please say so politely."""

        try:
            from openai.types.chat import (
                ChatCompletionSystemMessageParam,
                ChatCompletionUserMessageParam
            )

            messages: List[ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": response_prompt}
            ]

            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1
            )

            final_response = response.choices[0].message.content.strip()

            # Calculate confidence based on search scores
            avg_score = sum(r.final_score for r in search_results) / len(search_results)
            confidence = min(avg_score * 2, 1.0)

            print(
                f"🎯 Generated response in {self.multilingual_handler.supported_languages[user_language]}: {final_response[:100]}...")

            return {
                "response": final_response,
                "confidence": confidence,
                "sources": sources,
                "language": user_language
            }

        except Exception as e:
            print(f"Error generating multilingual response: {e}")
            fallback_response = self.multilingual_handler.get_fallback_response(user_language)
            return {
                "response": fallback_response,
                "confidence": 0.0,
                "sources": [],
                "language": user_language
            }

    def generate_response_with_context(self, query: str, conversation_context: str = "", conversation_summary: str = None, max_tokens: int = 200) -> Dict[str, Any]:
        """
        Generate a response using both the current query and previous conversation context.
        """
        # Detect language from the latest user query only
        user_language = self.multilingual_handler.detect_language(query)

        # Combine context and summary with the query
        full_context = ""
        if conversation_summary:
            full_context += f"Conversation Summary: {conversation_summary}\n"
        if conversation_context:
            full_context += f"Recent history:\n{conversation_context}\n"
        full_context += f"User question: {query}"

        # Pass user_language to generate_response
        return self.generate_response(full_context, max_tokens=max_tokens, user_language=user_language)

    def initialize_knowledge_base(self):
        """Smart initialization - load existing or build new"""

        # Try to load existing database first
        if self._load_existing_database():
            print("✅ Loaded existing multilingual knowledge base")
            self.is_initialized = True
            return True

        # Build new database if none exists
        print("🔄 Building new multilingual knowledge base...")
        documents = load_multilingual_documents_from_folder("./documents")

        if not documents:
            print("❌ No documents found in ./documents folder")
            self.is_initialized = False
            return False

        self.build_knowledge_base(documents)
        self._save_metadata()
        self.is_initialized = True
        print("✅ Multilingual knowledge base built and saved")
        return True

    def _load_existing_database(self) -> bool:
        """Load existing Chroma database and rebuild multilingual BM25"""
        try:
            if not os.path.exists(self.vector_db.persist_directory):
                print("📁 No existing database found")
                return False

            self.vector_db.vectorstore = Chroma(
                persist_directory=self.vector_db.persist_directory,
                embedding_function=self.vector_db.embeddings
            )

            try:
                all_docs = self.vector_db.vectorstore.get()
                if not all_docs.get('documents') or len(all_docs['documents']) == 0:
                    print("📁 Database exists but is empty")
                    return False

                print(f"📁 Found {len(all_docs['documents'])} documents in database")

            except Exception as e:
                print(f"📁 Error reading from database: {e}")
                return False

            # Rebuild chunks from vectorstore data
            self.vector_db.chunks = []
            for i, (doc_content, metadata) in enumerate(zip(all_docs['documents'], all_docs['metadatas'])):
                chunk = ContextualChunk(
                    id=metadata.get('chunk_id', f'chunk_{i}'),
                    original_content=metadata.get('original_content', ''),
                    context=metadata.get('context', ''),
                    combined_content=doc_content,
                    embedding=None,
                    doc_id=metadata.get('doc_id', ''),
                    language=metadata.get('language', 'en'),
                    translated_content=metadata.get('translated_content', ''),
                    metadata=metadata
                )
                self.vector_db.chunks.append(chunk)

            # Rebuild multilingual BM25 index
            self.bm25.fit(self.vector_db.chunks)

            print(f"📁 Rebuilt multilingual BM25 index with {len(self.vector_db.chunks)} chunks")
            return True

        except Exception as e:
            print(f"❌ Error loading existing database: {e}")
            return False

    def _save_metadata(self):
        """Save metadata about the multilingual knowledge base"""
        # Count chunks by language
        language_counts = {}
        for chunk in self.vector_db.chunks:
            lang = chunk.language
            language_counts[lang] = language_counts.get(lang, 0) + 1

        metadata = {
            "created_at": datetime.now().isoformat(),
            "total_chunks": len(self.vector_db.chunks),
            "language_counts": language_counts,
            "languages_supported": list(self.multilingual_handler.supported_languages.keys()),
            "supported_language_names": self.multilingual_handler.supported_languages,
            "model_info": {
                "llm_model": self.llm_model,
                "embedding_model": self.embedding_model,
                "api_version": self.api_version
            }
        }

        metadata_file = os.path.join(self.vector_db.persist_directory, "multilingual_kb_metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"📊 Metadata saved: {metadata['total_chunks']} chunks across {len(language_counts)} languages")
        for lang, count in language_counts.items():
            lang_name = self.multilingual_handler.supported_languages.get(lang, lang.upper())
            print(f"   - {lang_name} ({lang}): {count} chunks")


    def chat(self, message: str) -> str:
        """Simple multilingual chat interface"""
        result = self.generate_response(message)
        return result["response"]

    # For backward compatibility, alias the new class
    def save_knowledge_base(self, filepath: str):
        """Save the knowledge base to disk"""
        if self.vector_db.vectorstore:
            self.vector_db.vectorstore.persist()

        # Save multilingual metadata
        self._save_metadata()
        print(f"Multilingual knowledge base saved to {filepath}")

    def load_knowledge_base(self) -> bool:
        """Load knowledge base from disk"""
        success = self.vector_db.load_existing()
        if success and self.vector_db.chunks:
            self.bm25.fit(self.vector_db.chunks)
            print(f"Multilingual knowledge base loaded from {self.vector_db.persist_directory}")
            return True
        return False


def load_multilingual_documents_from_folder(folder_path: str = "./documents") -> List[Document]:
    """Load documents from folder with language detection"""
    documents = []

    if not os.path.exists(folder_path):
        print(f"Documents folder not found: {folder_path}")
        return documents

    try:
        # Use LangChain's DirectoryLoader
        loader = DirectoryLoader(
            folder_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )

        langchain_docs = loader.load()

        # Initialize multilingual handler for language detection
        multilingual_handler = MultilingualHandler()

        # Convert to multilingual Document format
        for i, doc in enumerate(langchain_docs):
            file_path = Path(doc.metadata.get('source', f'doc_{i}'))
            doc_id = file_path.stem

            # Detect document language
            detected_language = multilingual_handler.detect_language(doc.page_content)

            document = Document(
                id=doc_id,
                content=doc.page_content,
                language=detected_language,
                metadata={
                    "source": doc.metadata.get('source', ''),
                    "type": "document",
                    "category": "customer_service",
                    "file_name": file_path.name,
                    "language": detected_language,
                    "character_count": len(doc.page_content)
                },
                created_at=datetime.now()
            )
            documents.append(document)
            print(f"Loaded document: {doc_id} ({detected_language}, {len(doc.page_content)} characters)")

        print(f"Successfully loaded {len(documents)} multilingual documents from {folder_path}")

        # Print language distribution
        language_counts = {}
        for doc in documents:
            lang = doc.language
            language_counts[lang] = language_counts.get(lang, 0) + 1

        print("Language distribution:")
        for lang, count in language_counts.items():
            lang_name = {'en': 'English', 'tl': 'Tagalog', 'ceb': 'Cebuano'}.get(lang, lang)
            print(f"  {lang_name}: {count} documents")

        return documents

    except Exception as e:
        print(f"Error loading multilingual documents from {folder_path}: {e}")
        return documents


# Create alias for backward compatibility
RAGChatbot = MultilingualRAGChatbot

def main():
    """Main function for development - initializes multilingual chatbot"""
    print("🚀 Initializing Multilingual RAG Chatbot...")
    print("🌍 Supporting English, Tagalog, and Cebuano")
    print("=" * 60)

    # Initialize multilingual chatbot
    chatbot = MultilingualRAGChatbot()

    if not chatbot.is_initialized:
        print("❌ Failed to initialize multilingual knowledge base")
        print("Make sure documents are available in ./documents folder")
        return

    print("✅ Multilingual RAG Chatbot ready!")
    print("🗣️  Supported languages: English, Tagalog, Cebuano")
    print("🔍 Features: Language detection, translation, cross-language search")


if __name__ == "__main__":
    main()
