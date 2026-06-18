# retriever.py
"""
Advanced retrieval system with hybrid search and re-ranking
"""
import numpy as np
from typing import List, Optional
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


class AdvancedRetriever:
    """
    Advanced retrieval combining:
    - Semantic (vector) search
    - BM25 keyword search
    - Re-ranking
    - Category filtering
    """
    
    def __init__(self, brain):
        self.brain = brain
        self.embeddings = brain.embeddings
        self.vectorstore = brain.vectorstore
        self.bm25_index = None
        self.bm25_docs = None
        
        # Build BM25 index
        self._build_bm25_index()
    
    def _build_bm25_index(self):
        """Build BM25 index from all documents in the brain"""
        try:
            all_data = self.vectorstore.get()
            
            if all_data and all_data.get('documents'):
                documents = all_data['documents']
                metadatas = all_data.get('metadatas', [])
                
                # Tokenize for BM25
                tokenized_corpus = [
                    doc.lower().split() 
                    for doc in documents
                ]
                
                self.bm25_index = BM25Okapi(tokenized_corpus)
                self.bm25_docs = list(zip(documents, metadatas))
                print(f"🔍 BM25 index built: {len(documents)} documents")
        except Exception as e:
            print(f"⚠️  Warning: Could not build BM25 index: {e}")
            self.bm25_index = None
    
    def hybrid_search(
        self,
        query: str,
        k: int = 4,
        category: str = None,
        hybrid_k: int = 10
    ) -> List[Document]:
        """
        Hybrid search combining vector + BM25 + category filtering
        """
        
        # Build filter for category
        filter_dict = None
        if category and category != "all":
            filter_dict = {"category": category}
        
        # Vector search
        vector_docs = self.vectorstore.similarity_search(
            query, 
            k=k * 2,
            filter=filter_dict
        )
        
        # BM25 search
        bm25_docs = []
        if self.bm25_index:
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25_index.get_scores(tokenized_query)
            
            # Get top scoring documents
            top_indices = np.argsort(bm25_scores)[-k * 2:][::-1]
            
            for idx in top_indices:
                if bm25_scores[idx] > 0:
                    doc_text, metadata = self.bm25_docs[idx]
                    
                    # Apply category filter
                    if filter_dict:
                        if metadata.get("category") != category:
                            continue
                    
                    bm25_docs.append(Document(
                        page_content=doc_text,
                        metadata=metadata
                    ))
        
        # Combine results (deduplicate)
        seen_content = set()
        combined = []
        
        # Add vector results first
        for doc in vector_docs:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_content:
                combined.append(doc)
                seen_content.add(content_hash)
        
        # Add BM25 results
        for doc in bm25_docs:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_content:
                combined.append(doc)
                seen_content.add(content_hash)
        
        return combined[:hybrid_k]
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 4
    ) -> List[Document]:
        """
        Re-rank documents using cosine similarity with query embedding
        """
        if not documents:
            return []
        
        # Get query embedding
        query_embedding = np.array(self.embeddings.embed_query(query))
        
        # Score each document
        scored_docs = []
        for doc in documents:
            # Get document embedding
            doc_embedding = np.array(self.embeddings.embed_query(doc.page_content))
            
            # Cosine similarity
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            
            scored_docs.append((similarity, doc))
        
        # Sort by similarity (highest first)
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        return [doc for score, doc in scored_docs[:top_k]]
    
    def retrieve(
        self,
        query: str,
        k: int = 4,
        category: str = None,
        use_hybrid: bool = True,
        use_rerank: bool = True
    ) -> List[Document]:
        """
        Complete retrieval pipeline
        """
        
        if use_hybrid:
            docs = self.hybrid_search(query, k=k, category=category, hybrid_k=10)
        else:
            filter_dict = None
            if category and category != "all":
                filter_dict = {"category": category}
            docs = self.vectorstore.similarity_search(query, k=k*2, filter=filter_dict)
        
        if use_rerank and len(docs) > k:
            docs = self.rerank(query, docs, top_k=k)
        
        return docs[:k]