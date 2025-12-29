from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from pymongo import MongoClient
import os
from dotenv import load_dotenv


load_dotenv()

class DocumentProcessor:
    def __init__(self):
        # Initialize Azure OpenAI Embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("embedding_AZURE_OPENAI_API_BASE"),
            api_key=os.getenv("embedding_AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("embedding_AZURE_OPENAI_API_VERSION"),
            azure_deployment=os.getenv("embedding_AZURE_OPENAI_API_NAME")
        )
        
        # Initialize MongoDB
        self.mongo_client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017/"))
        self.db = self.mongo_client['ai_tutor']
        self.embeddings_collection = self.db['embeddings']
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=400,
            length_function=len
        )
    
    def process_document(self, file_path, user_id, material_id):
        """Process document and store embeddings in MongoDB"""
        try:
            # Load document based on file type
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path)
            else:
                raise ValueError("Unsupported file format")
            
            # Load and split documents
            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)
            
            # Generate embeddings and store
            stored_count = 0
            for idx, chunk in enumerate(chunks):
                # Generate embedding
                embedding = self.embeddings.embed_query(chunk.page_content)
                
                # Store in MongoDB
                doc = {
                    'user_id': user_id,
                    'material_id': material_id,
                    'chunk_index': idx,
                    'content': chunk.page_content,
                    'embedding': embedding,
                    'metadata': chunk.metadata
                }
                self.embeddings_collection.insert_one(doc)
                stored_count += 1
            
            return {
                'success': True,
                'chunks_processed': stored_count,
                'message': f'Successfully processed {stored_count} chunks'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_relevant_chunks(self, query, user_id, material_id=None, use_all_materials=False, top_k=5):
        """Retrieve relevant chunks using similarity search"""
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Build MongoDB filter
            filter_query = {'user_id': user_id}
            if not use_all_materials and material_id:
                filter_query['material_id'] = material_id
            
            # Get all documents matching filter
            documents = list(self.embeddings_collection.find(filter_query))
            
            if not documents:
                return []
            
            # Calculate cosine similarity
            from numpy import dot
            from numpy.linalg import norm
            
            def cosine_similarity(a, b):
                return dot(a, b) / (norm(a) * norm(b))
            
            # Calculate similarities
            similarities = []
            for doc in documents:
                similarity = cosine_similarity(query_embedding, doc['embedding'])
                similarities.append({
                    'content': doc['content'],
                    'similarity': similarity,
                    'metadata': doc.get('metadata', {})
                })
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            print(f"Error retrieving chunks: {str(e)}")
            return []