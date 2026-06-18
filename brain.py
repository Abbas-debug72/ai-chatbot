# brain.py – FINAL PRODUCTION VERSION (with strict threshold filtering)
import os
import json
import re
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from collections import Counter

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class KnowledgeBrain:
    """Universal knowledge engine for any document collection."""
    
    def __init__(self, pdf_directory="./pdfs", persist_directory="./vector_store",
                 embedding_model="all-MiniLM-L6-v2", chunk_size=800, chunk_overlap=150):
        
        self.pdf_directory = Path(pdf_directory)
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        print("📥 Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'trust_remote_code': True}
        )
        
        os.makedirs(persist_directory, exist_ok=True)
        self.index_path = os.path.join(persist_directory, "faiss_index")
        
        print("📂 Loading vector store...")
        if os.path.exists(self.index_path):
            self.vectorstore = FAISS.load_local(
                self.index_path, self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("   ✅ Loaded existing index")
        else:
            self.vectorstore = None
            print("   ℹ️  No existing index found")
        
        self.metadata_file = os.path.join(persist_directory, "brain_metadata.json")
        self.documents_metadata = self._load_metadata()
        self.doc_summaries = self._build_document_summaries()
        
        stats = self.get_stats()
        print(f"✅ Brain ready: {stats['total_documents']} docs, {stats['total_chunks']} chunks\n")
    
    # ============================================================
    # METADATA
    # ============================================================
    
    def _load_metadata(self) -> Dict:
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_metadata(self):
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.documents_metadata, f, indent=2, default=str)
    
    # ============================================================
    # DOCUMENT ANALYSIS
    # ============================================================
    
    def _build_document_summaries(self) -> Dict:
        summaries = {}
        if not self.vectorstore:
            return summaries
        
        for filename in self.documents_metadata:
            chunks = self.search_by_filename(filename, k=100)
            if not chunks:
                continue
            
            full_text = " ".join([c.page_content for c in chunks])
            
            summaries[filename] = {
                "total_chars": len(full_text),
                "main_entities": self._extract_entities(full_text),
                "key_terms": self._extract_key_terms(full_text),
                "document_type": self._detect_document_type(full_text, filename),
            }
        
        return summaries
    
    def _extract_entities(self, text: str) -> List[str]:
        entities = []
        patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'([A-Z][a-z]+\s+(?:University|College|Institute|Company|Corp|Inc|Ltd|Hospital|Center)[^\.,]*)',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches)
        entity_counts = Counter(entities)
        return [e for e, c in entity_counts.most_common(15)]
    
    def _extract_key_terms(self, text: str, max_terms: int = 25) -> List[str]:
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        stop_words = {
            'this', 'that', 'with', 'from', 'they', 'have', 'been', 'were',
            'about', 'which', 'their', 'there', 'would', 'could', 'should',
            'these', 'those', 'what', 'when', 'where', 'over', 'into', 'also',
            'after', 'before', 'between', 'under', 'above', 'each', 'every',
            'other', 'some', 'such', 'only', 'then', 'than', 'just', 'because',
            'through', 'during', 'being', 'having', 'more', 'most', 'very'
        }
        filtered = [w for w in words if w not in stop_words]
        term_counts = Counter(filtered)
        return [term for term, count in term_counts.most_common(max_terms)]
    
    def _detect_document_type(self, text: str, filename: str) -> str:
        """Detect document type from content AND filename."""
        text_lower = text.lower()[:3000]
        filename_lower = filename.lower()
        
        # Check filename first (strongest signal)
        if any(w in filename_lower for w in ['tonsil', 'surgery', 'medical', 'health', 'clinical', 'patient']):
            return "medical"
        if any(w in filename_lower for w in ['cv', 'resume']):
            return "resume/cv"
        if any(w in filename_lower for w in ['transcript', 'proforma', 'degree', 'guideline']):
            return "academic_record"
        
        # Then check content
        if re.search(r'\b(tonsil|tonsillectomy|surgery|surgical|gland|adenoid|anesthesia|post.op)', text_lower):
            return "medical"
        if re.search(r'(curriculum\s*vitae|\bcv\b|objective|experience|certification|linkedin|github)', text_lower):
            return "resume/cv"
        if re.search(r'(transcript|proforma|deposit\s*slip|clearance\s*certificate|matric)', text_lower):
            return "academic_record"
        if re.search(r'(guideline|procedure|step\s*\d|instruction|manual)', text_lower):
            return "guide/manual"
        
        return self._detect_category(filename)
    
    def _detect_category(self, filename: str) -> str:
        filename_lower = filename.lower()
        categories = {
            "resume": ["cv", "resume", "biodata"],
            "technical": ["tech", "code", "programming", "software", "api", "devops", "sqa"],
            "business": ["business", "report", "financial", "annual"],
            "legal": ["legal", "contract", "agreement", "policy"],
            "guide": ["manual", "guide", "tutorial", "guideline"],
            "academic": ["academic", "course", "syllabus", "transcript", "degree", "proforma"],
            "medical": ["medical", "health", "tonsil", "surgery"],
        }
        for category, keywords in categories.items():
            if any(kw in filename_lower for kw in keywords):
                return category
        return "general"
    
    # ============================================================
    # INGESTION
    # ============================================================
    
    def ingest_all_pdfs(self) -> Dict:
        pdf_files = list(self.pdf_directory.glob("**/*.pdf"))
        
        if not pdf_files:
            print("❌ No PDFs found!")
            return {"total": 0, "processed": 0, "skipped": 0, "failed": 0}
        
        print(f"📚 Found {len(pdf_files)} PDFs\n")
        
        all_chunks = []
        results = {"total": len(pdf_files), "processed": 0, "skipped": 0, "failed": 0}
        
        for i, pdf_path in enumerate(pdf_files, 1):
            try:
                filename = pdf_path.name
                
                if filename in self.documents_metadata:
                    print(f"[{i}/{len(pdf_files)}] ⏭️  {filename}")
                    results["skipped"] += 1
                    continue
                
                print(f"[{i}/{len(pdf_files)}] 📄 {filename}...", end=" ", flush=True)
                
                loader = PyPDFLoader(str(pdf_path))
                pages = loader.load()
                
                if not pages:
                    print("❌ No content")
                    results["failed"] += 1
                    continue
                
                category = self._detect_category(filename)
                
                for page_num, page in enumerate(pages):
                    page.metadata.update({
                        "source_file": filename,
                        "category": category,
                        "page_number": page_num + 1,
                    })
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
                chunks = text_splitter.split_documents(pages)
                all_chunks.extend(chunks)
                
                self.documents_metadata[filename] = {
                    "pages": len(pages),
                    "chunks": len(chunks),
                    "category": category,
                    "upload_date": datetime.now().isoformat()
                }
                
                print(f"✅ {len(pages)}p → {len(chunks)} chunks [{category}]")
                results["processed"] += 1
                
                if results["processed"] % 10 == 0:
                    self._save_metadata()
                    
            except Exception as e:
                print(f"❌ {str(e)[:80]}")
                results["failed"] += 1
        
        if all_chunks:
            print(f"\n🔨 Building FAISS index...")
            if self.vectorstore is None:
                self.vectorstore = FAISS.from_documents(all_chunks, self.embeddings)
            else:
                self.vectorstore.add_documents(all_chunks)
            self.vectorstore.save_local(self.index_path)
            print("✅ Index saved!")
        
        self._save_metadata()
        self.doc_summaries = self._build_document_summaries()
        return results
    
    # ============================================================
    # SEARCH
    # ============================================================
    
    def search(self, query: str, k: int = 4, category: str = None) -> List[Document]:
        if not self.vectorstore:
            return []
        if category and category != "all":
            docs = self.vectorstore.similarity_search(query, k=k * 4)
            return [d for d in docs if d.metadata.get("category") == category][:k]
        return self.vectorstore.similarity_search(query, k=k)
    
    def intelligent_search(self, query: str, k: int = 4, category: str = None) -> List[Document]:
        """
        Intelligent search with strict relevance filtering.
        Returns only documents that score above a dynamic threshold.
        May return fewer than k results if not enough are relevant.
        """
        candidates = self.search(query, k=k * 5, category=category)
        
        if len(candidates) <= k:
            return candidates

        query_analysis = self._analyze_query(query)

        scored = []
        for doc in candidates:
            filename = doc.metadata.get('source_file', '')
            doc_info = self.doc_summaries.get(filename, {})
            doc_type = doc_info.get('document_type', 'unknown')

            filename_score = self._score_filename_match(query_analysis, filename)
            topic_score = self._score_topic_match(query_analysis, doc_type, doc_info)
            content_score = self._score_content_match(query_analysis, doc.page_content)
            entity_score = self._score_entity_match(query_analysis, doc_info)

            total = filename_score + topic_score + content_score + entity_score
            scored.append((total, doc))

        # Sort highest first
        scored.sort(key=lambda x: x[0], reverse=True)

        # Dynamic threshold: must be at least 50% of the best score, minimum 30
        if scored:
            top_score = scored[0][0]
            MIN_SCORE = max(30, top_score * 0.5)
        else:
            MIN_SCORE = 30

        # Debug output
        print(f"   🔍 '{query}' [{query_analysis.get('type', '?')}]")
        print(f"   📊 Threshold: {MIN_SCORE:.0f} (top score: {scored[0][0]:.0f})")
        for score, doc in scored[:6]:
            fname = doc.metadata.get('source_file', '?')[:45]
            flag = "✅" if score >= MIN_SCORE else "⏭️"
            print(f"     {score:5.0f} {flag} {fname}")

        # Collect unique sources that pass the threshold
        seen = set()
        result = []
        for score, doc in scored:
            source = doc.metadata.get('source_file', '')
            if score < MIN_SCORE:
                continue        # skip weak matches
            if source not in seen:
                seen.add(source)
                result.append(doc)

        # Fallback ONLY if absolutely nothing passed the threshold
        if len(result) == 0 and scored:
            print("   ⚠️  No documents above threshold – falling back to top 2")
            for score, doc in scored[:2]:
                source = doc.metadata.get('source_file', '')
                if source not in seen:
                    seen.add(source)
                    result.append(doc)

        print(f"   ✅ Returning {len(result)} documents")
        return result[:k]
    
    def _analyze_query(self, query: str) -> Dict:
        """Analyze query to understand intent, extract entities and key terms."""
        # Clean the query: remove punctuation
        query_clean = re.sub(r'[?.,!;:()"\'*]', ' ', query)
        query_clean = re.sub(r'\s+', ' ', query_clean).strip()
        
        query_lower = query_clean.lower()
        query_words = set(query_lower.split())
        
        analysis = {
            "original": query,
            "words": query_words,
            "entities": [],
            "type": "general",
            "key_terms": set(),
        }
        
        # Extract capitalized words as potential named entities
        names = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', query_clean)
        analysis["entities"] = [n.lower() for n in names]
        
        # Detect query type
        if any(w in query_lower for w in ['who', 'whose', 'person', 'he', 'she', 'his', 'her']):
            analysis["type"] = "person_query"
        elif any(w in query_lower for w in ['how', 'procedure', 'process', 'steps', 'guide']):
            analysis["type"] = "how_to_query"
        elif any(w in query_lower for w in ['what', 'define', 'explain', 'meaning']):
            analysis["type"] = "definition_query"
        
        # Key terms (remove stop words)
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'do', 'does',
            'did', 'has', 'have', 'he', 'she', 'it', 'they', 'him', 'her',
            'his', 'what', 'who', 'where', 'when', 'why', 'how', 'check',
            'tell', 'show', 'find', 'know', 'about', 'me', 'you', 'can',
            'could', 'would', 'should', 'and', 'or', 'but', 'if', 'of',
            'at', 'by', 'for', 'with', 'from', 'to', 'in', 'on', 'this',
            'get', 'got', 'does', 'any', 'some', 'the', 'i', 'my', 'mine',
            'that', 'these', 'those', 'then', 'than', 'just', 'also',
            'very', 'much', 'such', 'only', 'over', 'into', 'after'
        }
        
        analysis["key_terms"] = {w for w in query_words if w not in stop_words and len(w) > 1}
        
        return analysis
    
    def _score_filename_match(self, qa: Dict, filename: str) -> float:
        """Score based on query terms appearing in filename."""
        fn = filename.lower()
        score = 0
        
        # Named entities in filename = strong signal
        for entity in qa.get('entities', []):
            if entity in fn:
                score += 80
        
        # Key terms in filename = good signal
        for term in qa.get('key_terms', []):
            if term in fn:
                score += 60
        
        return score
    
    def _score_topic_match(self, qa: Dict, doc_type: str, doc_info: Dict) -> float:
        """Score based on document type matching query intent."""
        query_type = qa.get('type', 'general')
        query_terms = qa.get('key_terms', set())
        score = 0
        
        # Check term overlap with document key terms
        doc_key_terms = set(doc_info.get('key_terms', []))
        term_overlap = len(query_terms & doc_key_terms)
        
        if query_type == "person_query":
            if doc_type == "resume/cv":
                score += 80
            elif doc_type == "medical":
                score -= 80  # Medical docs are NOT about people
            elif doc_type in ["guide/manual", "academic_record"]:
                score -= 50
        
        elif query_type == "how_to_query":
            if doc_type in ["guide/manual", "academic_record"]:
                score += 80
            elif doc_type == "resume/cv":
                score -= 60
            elif doc_type == "medical":
                score -= 40
        
        elif query_type == "definition_query":
            # Bonus for term overlap
            if term_overlap >= 3:
                score += 80
            elif term_overlap >= 1:
                score += 40
            else:
                score -= 20
            
            # Topic-specific bonuses
            medical_terms = {'medical', 'surgery', 'disease', 'tonsil', 'tonsillectomy', 
                           'treatment', 'diagnosis', 'clinical', 'patient', 'health'}
            cv_terms = {'cv', 'resume', 'job', 'work', 'experience', 'skills'}
            academic_terms = {'transcript', 'degree', 'form', 'proforma', 'guideline', 'issuance'}
            
            if query_terms & medical_terms and doc_type == 'medical':
                score += 50
            if query_terms & cv_terms and doc_type == 'resume/cv':
                score += 50
            if query_terms & academic_terms and doc_type in ['academic_record', 'guide/manual']:
                score += 50
        
        return score
    
    def _score_content_match(self, qa: Dict, content: str) -> float:
        """Score based on query terms appearing in document content."""
        content_lower = content.lower()
        score = 0
        
        # Entity matches
        for entity in qa.get('entities', []):
            if entity in content_lower:
                score += 20
        
        # Key term density
        term_count = sum(1 for t in qa.get('key_terms', []) if t in content_lower)
        score += term_count * 3
        
        return score
    
    def _score_entity_match(self, qa: Dict, doc_info: Dict) -> float:
        """Score based on query entities matching document entities."""
        if not doc_info:
            return 0
        
        score = 0
        doc_entities = [e.lower() for e in doc_info.get('main_entities', [])]
        
        # Check extracted named entities
        for entity in qa.get('entities', []):
            if any(entity in de for de in doc_entities):
                score += 30
        
        # Also check key terms against document entities
        for term in qa.get('key_terms', []):
            if any(term in de for de in doc_entities):
                score += 25
        
        return score
    
    # ============================================================
    # UTILITIES
    # ============================================================
    
    def search_by_filename(self, filename: str, k: int = 4) -> List[Document]:
        if not self.vectorstore:
            return []
        docs = self.vectorstore.similarity_search(filename, k=k * 4)
        return [d for d in docs if d.metadata.get("source_file") == filename][:k]
    
    def get_stats(self) -> Dict:
        total_chunks = 0
        if self.vectorstore:
            try:
                total_chunks = self.vectorstore.index.ntotal
            except:
                pass
        
        cats = {}
        for m in self.documents_metadata.values():
            cat = m.get("category", "general")
            cats[cat] = cats.get(cat, 0) + 1
        
        size = 0
        if os.path.exists(self.persist_directory):
            for dp, dn, fn in os.walk(self.persist_directory):
                for f in fn:
                    fp = os.path.join(dp, f)
                    if os.path.exists(fp):
                        size += os.path.getsize(fp)
        
        return {
            "total_documents": len(self.documents_metadata),
            "total_chunks": total_chunks,
            "total_pages": sum(m.get("pages", 0) for m in self.documents_metadata.values()),
            "categories": cats,
            "brain_size_mb": round(size / (1024 * 1024), 2),
        }
    
    def get_categories(self) -> List[str]:
        return sorted(set(m.get("category", "general") for m in self.documents_metadata.values()))
    
    def get_all_filenames(self) -> List[str]:
        return list(self.documents_metadata.keys())