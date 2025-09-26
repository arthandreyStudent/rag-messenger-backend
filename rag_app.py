import os
import json
import hashlib
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Core dependencies
from openai import AzureOpenAI
from typing import cast, Any
from dotenv import load_dotenv

# LangChain dependencies - Fixed imports
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI  # Fixed: Updated import
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.schema import Document as LangChainDocument
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Traditional search dependencies
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data - Fixed typo
try:
    nltk.download('punkt', quiet=True)  # Fixed: was 'punk'
    nltk.download('stopwords', quiet=True)
except:
    pass

load_dotenv()

@dataclass
class Document:
    """Document structure for RAG system"""
    id: str
    content: str
    metadata: Dict[str, Any]
    created_at: datetime

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

@dataclass
class SearchResult:
    """Search result with relevance scores"""
    chunk: ContextualChunk
    vector_score: float
    bm25_score: float
    rerank_score: float
    final_score: float


def _get_cache_key(prompt: str) -> str:
    return hashlib.md5(prompt.encode()).hexdigest()


class PromptCache:
    """Simple in-memory prompt caching system"""

    def __init__(self, ttl_hours: int = 24):
        self.cache = {}
        self.ttl_hours = ttl_hours

    def get(self, prompt: str) -> Optional[str]:
        key = _get_cache_key(prompt)
        if key in self.cache:
            response, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(hours=self.ttl_hours):
                return response
            else:
                del self.cache[key]
        return None

    def set(self, prompt: str, response: str):
        key = _get_cache_key(prompt)
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
    """Chroma vector database class using LangChain integration"""

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory

        # Initialize Azure OpenAI embeddings using environment variables - Fixed configuration
        self.embeddings = AzureOpenAIEmbeddings(
            model=os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_version="2024-12-01-preview",
            chunk_size=1  # Fixed: Added required chunk_size parameter
        )

        self.vectorstore = None
        self.chunks = []

        # Ensure the persist directory exists
        os.makedirs(persist_directory, exist_ok=True)

    def add_chunks(self, chunks: List[ContextualChunk]):
        """Add contextual chunks to Chroma vector database"""
        if not chunks:
            return

        self.chunks = chunks

        # Prepare texts and metadatas for Chroma
        texts = [chunk.combined_content for chunk in chunks]
        metadata = [
            {
                "chunk_id": chunk.id,
                "doc_id": chunk.doc_id,
                "original_content": chunk.original_content[:500],  # Truncate for metadata
                "context": chunk.context,
                **chunk.metadata
            }
            for chunk in chunks
        ]

        # Create Chroma vectorstore
        self.vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadata,
            persist_directory=self.persist_directory
        )

        # Persist the database
        self.vectorstore.persist()

    def search(self, query: str, k: int = 10) -> List[Tuple[ContextualChunk, float]]:
        """Search for similar vectors using semantic similarity"""
        if not self.vectorstore or not self.chunks:
            return []

        try:
            # Perform similarity search with scores
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

class ContextualBM25:
    """BM25 implementation for contextual chunks"""

    def __init__(self):
        self.bm25 = None
        self.chunks = []
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()  # Fallback if NLTK data not available

    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for BM25"""
        try:
            tokens = word_tokenize(text.lower())
            return [token for token in tokens if token.isalnum() and token not in self.stop_words]
        except:
            # Fallback if NLTK fails
            return [word.lower() for word in text.split() if word.isalnum()]

    def fit(self, chunks: List[ContextualChunk]):
        """Fit BM25 on contextual chunks"""
        self.chunks = chunks
        corpus = []

        for chunk in chunks:
            # Use combined content (context + original) for BM25
            processed_text = self._preprocess_text(chunk.combined_content)
            corpus.append(processed_text)

        if corpus:
            self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int = 10) -> List[Tuple[ContextualChunk, float]]:
        """Search using BM25"""
        if not self.bm25 or not self.chunks:
            return []

        query_tokens = self._preprocess_text(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Get top k results
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only return results with positive scores
                results.append((self.chunks[idx], float(scores[idx])))

        return results

class RAGChatbot:
    """Main RAG-based AI Chatbot class"""

    def __init__(self,
                 azure_api_key: str = None,
                 azure_endpoint: str = None,
                 api_version: str = "2024-12-01-preview",
                 deployment_name: str = None,
                 embedding_deployment: str = None):

        # Azure configuration
        self.azure_api_key = azure_api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", "https://geral-mcul2aqz-eastus2.cognitiveservices.azure.com/")
        self.api_version = api_version
        self.deployment_name = deployment_name or os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-35-turbo")
        self.embedding_deployment = embedding_deployment or os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
            api_key=self.azure_api_key
        )

        # Initialize ChromaVectorDB
        self.vector_db = ChromaVectorDB()
        self.bm25 = ContextualBM25()
        self.prompt_cache = PromptCache()

        # Initialize LangChain Azure components
        self.embeddings = AzureOpenAIEmbeddings(
            model=self.embedding_deployment,
            azure_endpoint=self.azure_endpoint,
            api_key=self.azure_api_key,
            openai_api_version=self.api_version,
            chunk_size=1
        )

        self.llm = AzureChatOpenAI(
            deployment_name=self.deployment_name,
            openai_api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
            openai_api_key=self.azure_api_key,
            temperature=0.1
        )

        # Initialize text splitter with optimal settings for contextual chunks
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

        # System prompt for customer service
        self.system_prompt = """You are a helpful customer service assistant for a small-medium enterprise (SME) business. 
        You handle customer inquiries via Facebook Messenger with a friendly, professional tone.
        
        Guidelines:
        - Be concise but helpful
        - Use the provided context to answer questions accurately
        - If you cannot find relevant information in the context, say so politely
        - Maintain a conversational, approachable tone
        - Focus on solving customer problems quickly"""

        # Initialize knowledge base
        self.is_initialized = False
        self.initialize_knowledge_base()

    def initialize_knowledge_base(self):
        """Smart initialization - load existing or build new"""

        # Try to load existing database first
        if self._load_existing_database():
            print("✅ Loaded existing knowledge base")
            self.is_initialized = True
            return True

        # Build new database if none exists
        print("🔄 Building new knowledge base...")
        documents = load_documents_from_folder("./documents")

        if not documents:
            print("❌ No documents found in ./documents folder")
            self.is_initialized = False
            return False

        self.build_knowledge_base(documents)
        self._save_metadata()
        self.is_initialized = True
        print("✅ Knowledge base built and saved")
        return True

    def _load_existing_database(self) -> bool:
        """Load existing Chroma database and rebuild BM25"""
        try:
            # Check if Chroma database exists
            if not os.path.exists(self.vector_db.persist_directory):
                print("📁 No existing database found")
                return False

            # Try to load Chroma vectorstore
            self.vector_db.vectorstore = Chroma(
                persist_directory=self.vector_db.persist_directory,
                embedding_function=self.vector_db.embeddings
            )

            # Get documents from vectorstore to check if it's populated
            try:
                doc_count = self.vector_db.vectorstore._collection.count()
                if doc_count == 0:
                    print("📁 Database exists but is empty")
                    return False
            except:
                print("📁 Database exists but appears corrupted")
                return False

            # Get all documents from the collection to rebuild chunks
            all_docs = self.vector_db.vectorstore.get()

            if not all_docs['documents']:
                print("📁 No documents in existing database")
                return False

            print(f"📁 Found {len(all_docs['documents'])} documents in database")

            # Rebuild chunks from vectorstore data
            self.vector_db.chunks = []
            for i, (doc_content, metadata) in enumerate(zip(all_docs['documents'], all_docs['metadatas'])):
                chunk = ContextualChunk(
                    id=metadata.get('chunk_id', f'chunk_{i}'),
                    original_content=metadata.get('original_content', ''),
                    context=metadata.get('context', ''),
                    combined_content=doc_content,
                    embedding=None,  # We don't need to store embeddings in memory
                    doc_id=metadata.get('doc_id', ''),
                    metadata=metadata
                )
                self.vector_db.chunks.append(chunk)

            # Rebuild BM25 index from chunks
            self.bm25.fit(self.vector_db.chunks)

            print(f"📁 Rebuilt BM25 index with {len(self.vector_db.chunks)} chunks")
            return True

        except Exception as e:
            print(f"❌ Error loading existing database: {e}")
            return False

    def _save_metadata(self):
        """Save metadata about the knowledge base"""
        metadata = {
            "chunks_count": len(self.vector_db.chunks),
            "last_build_time": datetime.now().isoformat(),
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "documents_folder": "./documents"
        }

        metadata_file = os.path.join(self.vector_db.persist_directory, "kb_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI"""
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return []

    def generate_context_for_chunk(self, chunk: str, document: str) -> str:
        """Generate contextual information for a chunk using LangChain ChatOpenAI"""
        prompt = f"""Please provide a brief, informative context for the following text chunk from a customer service document.
        
The context should explain what this chunk is about and how it relates to the overall document in 1-2 sentences.

Full document (excerpt): {document[:2000]}...

Text chunk: {chunk}

Context:"""

        # Check cache first
        cached_response = self.prompt_cache.get(prompt)
        if cached_response:
            return cached_response

        try:
            # Use LangChain ChatOpenAI with .invoke() instead of deprecated __call__
            from langchain.schema import HumanMessage, SystemMessage

            messages = [
                SystemMessage(content="You are an expert at creating concise, informative contexts for text chunks."),
                HumanMessage(content=prompt)
            ]

            response = self.llm.invoke(messages)  # Fixed: Use .invoke() instead of __call__
            context = response.content.strip()

            # Cache the response
            self.prompt_cache.set(prompt, context)

            return context

        except Exception as e:
            print(f"Error generating context: {e}")
            return "This chunk contains customer service information."

    def create_contextual_chunks(self, documents: List[Document]) -> List[ContextualChunk]:
        """Create contextual chunks from documents using LangChain RecursiveCharacterTextSplitter"""
        contextual_chunks = []

        for doc in documents:
            print(f"Processing document: {doc.id}")

            # Convert to LangChain document format
            langchain_doc = LangChainDocument(
                page_content=doc.content,
                metadata=doc.metadata
            )

            # Use RecursiveCharacterTextSplitter to split the document
            chunks = self.text_splitter.split_documents([langchain_doc])

            print(f"Split document {doc.id} into {len(chunks)} chunks")

            # Generate contextual information for each chunk
            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)} for document {doc.id}")

                # Generate context using LLM
                context = self.generate_context_for_chunk(chunk.page_content, doc.content)

                # Combine context with original chunk
                combined_content = f"Context: {context}\n\nContent: {chunk.page_content}"

                # Get embedding for combined content using LangChain embeddings
                try:
                    embedding = self.embeddings.embed_query(combined_content)
                except Exception as e:
                    print(f"Error getting embedding for chunk: {e}")
                    embedding = []

                # Create contextual chunk
                contextual_chunk = ContextualChunk(
                    id=f"{doc.id}_chunk_{i}",
                    original_content=chunk.page_content,
                    context=context,
                    combined_content=combined_content,
                    embedding=embedding,
                    doc_id=doc.id,
                    metadata={
                        **doc.metadata,
                        **chunk.metadata,
                        'chunk_index': i,
                        'chunk_size': len(chunk.page_content),
                        'chunk_start': chunk.metadata.get('start_index', 0)
                    }
                )

                contextual_chunks.append(contextual_chunk)

        return contextual_chunks

    def build_knowledge_base(self, documents: List[Document]):
        """Build knowledge base from documents"""
        print("Creating contextual chunks...")
        contextual_chunks = self.create_contextual_chunks(documents)

        print("Building vector database...")
        self.vector_db.add_chunks(contextual_chunks)

        print("Building BM25 index...")
        self.bm25.fit(contextual_chunks)

        print(f"Knowledge base built with {len(contextual_chunks)} contextual chunks")

    def rerank_results(self, query: str, results: List[SearchResult], top_k: int = 5) -> List[SearchResult]:
        """Rerank search results using LLM"""
        if len(results) <= top_k:
            return sorted(results, key=lambda x: x.final_score, reverse=True)

        # Prepare reranking prompt
        candidates = []
        for i, result in enumerate(results[:20]):  # Limit to top 20 for reranking
            candidates.append(f"[{i}] {result.chunk.combined_content[:500]}...")

        rerank_prompt = f"""Given the user query and the following text candidates, rank them by relevance to the query.
        Return only the indices of the top {top_k} most relevant candidates, in order of relevance.

Query: {query}

Candidates:
{chr(10).join(candidates)}

Top {top_k} most relevant indices (comma-separated):"""

        try:
            # Fixed: Properly typed messages for Azure OpenAI
            from openai.types.chat import (
                ChatCompletionSystemMessageParam,
                ChatCompletionUserMessageParam
            )

            messages: List[ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam] = [
                {"role": "system", "content": "You are an expert at ranking text relevance for search queries."},
                {"role": "user", "content": rerank_prompt}
            ]

            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.1,
                max_tokens=50
            )

            # Parse reranking results
            rerank_indices_str = response.choices[0].message.content.strip()
            rerank_indices = [int(x.strip()) for x in rerank_indices_str.split(',') if x.strip().isdigit()]

            # Apply reranking scores
            reranked_results = []
            for rank, idx in enumerate(rerank_indices[:top_k]):
                if idx < len(results):
                    result = results[idx]
                    result.rerank_score = (top_k - rank) / top_k  # Higher score for better rank
                    result.final_score = (result.vector_score + result.bm25_score + result.rerank_score) / 3
                    reranked_results.append(result)

            return reranked_results

        except Exception as e:
            print(f"Error in reranking: {e}")
            # Fallback to original ranking
            return sorted(results[:top_k], key=lambda x: x.final_score, reverse=True)

    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        """Hybrid search combining vector similarity and BM25"""
        # Vector search using Chroma
        vector_results = self.vector_db.search(query, k=20)

        # BM25 search
        bm25_results = self.bm25.search(query, k=20)

        # Combine results
        combined_results = {}

        # Add vector results
        for chunk, score in vector_results:
            combined_results[chunk.id] = SearchResult(
                chunk=chunk,
                vector_score=score,
                bm25_score=0.0,
                rerank_score=0.0,
                final_score=score
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
                    final_score=score
                )

        # Calculate combined scores
        results = list(combined_results.values())
        for result in results:
            # Simple weighted combination
            result.final_score = (0.6 * result.vector_score + 0.4 * result.bm25_score)

        # Sort by combined score
        results.sort(key=lambda x: x.final_score, reverse=True)

        # Rerank top results
        reranked_results = self.rerank_results(query, results, top_k=k)

        return reranked_results

    def generate_response(self, query: str, max_tokens: int = 500) -> Dict[str, Any]:
        """Generate response to user query"""
        # Search for relevant context
        search_results = self.search(query, k=5)

        if not search_results:
            return {
                "response": "I apologize, but I couldn't find relevant information to answer your question. Could you please rephrase or provide more details?",
                "sources": [],
                "confidence": 0.0
            }

        # Prepare context from search results
        context_parts = []
        sources = []

        for result in search_results:
            context_parts.append(result.chunk.combined_content)
            sources.append({
                "doc_id": result.chunk.doc_id,
                "chunk_id": result.chunk.id,
                "relevance_score": result.final_score
            })

        context = "\n\n---\n\n".join(context_parts)

        # Generate response
        response_prompt = f"""Context information:
{context}

User question: {query}

Please provide a helpful, accurate response based on the context above. If the context doesn't contain sufficient information to answer the question, please say so politely."""

        try:
            # Fixed: Properly typed messages for Azure OpenAI
            from openai.types.chat import (
                ChatCompletionSystemMessageParam,
                ChatCompletionUserMessageParam
            )

            messages: List[ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam] = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": response_prompt}
            ]

            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.7,
                max_tokens=max_tokens
            )

            generated_response = response.choices[0].message.content.strip()

            # Calculate confidence based on search scores
            avg_score = sum(r.final_score for r in search_results) / len(search_results)
            confidence = min(avg_score * 2, 1.0)  # Simple confidence calculation

            return {
                "response": generated_response,
                "sources": sources,
                "confidence": confidence
            }

        except Exception as e:
            print(f"Error generating response: {e}")
            return {
                "response": "I apologize, but I'm experiencing technical difficulties. Please try again later.",
                "sources": [],
                "confidence": 0.0
            }

    def chat(self, message: str) -> str:
        """Simple chat interface"""
        result = self.generate_response(message)
        return result["response"]

    def create_conversational_chain(self) -> ConversationalRetrievalChain:
        """Create LangChain conversational retrieval chain for advanced conversations"""
        if not self.vector_db.vectorstore:
            raise ValueError("Vector database not initialized. Build knowledge base first.")

        # Initialize conversation memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        # Create conversational retrieval chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_db.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            ),
            memory=memory,
            return_source_documents=True,
            verbose=True
        )

        return chain

    def chat_with_memory(self, message: str, chain: ConversationalRetrievalChain = None) -> Dict[str, Any]:
        """Chat with conversation memory using LangChain chain"""
        if chain is None:
            chain = self.create_conversational_chain()

        try:
            result = chain({"question": message})

            return {
                "response": result["answer"],
                "sources": [
                    {
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata
                    }
                    for doc in result.get("source_documents", [])
                ],
                "chat_history": result.get("chat_history", [])
            }
        except Exception as e:
            print(f"Error in conversational chat: {e}")
            return {
                "response": "I apologize, but I'm experiencing technical difficulties. Please try again later.",
                "sources": [],
                "chat_history": []
            }

    def save_knowledge_base(self, filepath: str):
        """Save the knowledge base to disk"""
        # Save Chroma vector database (it auto-persists)
        if self.vector_db.vectorstore:
            self.vector_db.vectorstore.persist()

        # Save additional metadata if needed
        metadata = {
            "chunks_count": len(self.vector_db.chunks),
            "created_at": datetime.now().isoformat(),
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model
        }

        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Knowledge base saved to {filepath}")

    def load_knowledge_base(self) -> bool:
        """Load knowledge base from disk"""
        success = self.vector_db.load_existing()
        if success and self.vector_db.chunks:
            # Rebuild BM25 index
            self.bm25.fit(self.vector_db.chunks)
            print(f"Knowledge base loaded from {self.vector_db.persist_directory}")
            return True
        return False

def load_documents_from_folder(folder_path: str = "./documents") -> List[Document]:
    """Load documents from the documents folder using LangChain"""
    documents = []

    if not os.path.exists(folder_path):
        print(f"Documents folder not found: {folder_path}")
        return documents

    try:
        # Use LangChain's DirectoryLoader to load text files
        loader = DirectoryLoader(
            folder_path,
            glob="**/*.txt",  # Load all .txt files recursively
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )

        langchain_docs = loader.load()

        # Convert LangChain documents to our Document format
        for i, doc in enumerate(langchain_docs):
            # Extract filename for document ID
            file_path = Path(doc.metadata.get('source', f'doc_{i}'))
            doc_id = file_path.stem  # Get filename without extension

            document = Document(
                id=doc_id,
                content=doc.page_content,
                metadata={
                    "source": doc.metadata.get('source', ''),
                    "type": "document",
                    "category": "customer_service",
                    "file_name": file_path.name
                },
                created_at=datetime.now()
            )
            documents.append(document)
            print(f"Loaded document: {doc_id} ({len(doc.page_content)} characters)")

        print(f"Successfully loaded {len(documents)} documents from {folder_path}")
        return documents

    except Exception as e:
        print(f"Error loading documents from {folder_path}: {e}")
        return documents

def main():
    """Main function for development - initializes chatbot and confirms it's ready"""
    print("🚀 Initializing RAG Chatbot...")
    print("=" * 50)

    # Initialize chatbot with smart database loading
    chatbot = RAGChatbot()

    if not chatbot.is_initialized:
        print("❌ Failed to initialize knowledge base")
        print("Make sure documents are available in ./documents folder")
        return

if __name__ == "__main__":
    main()
