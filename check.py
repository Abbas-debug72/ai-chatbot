# debug_scores.py
from brain import KnowledgeBrain

brain = KnowledgeBrain(pdf_directory="./pdfs", persist_directory="./vector_store")

# Test "who is abbas"
query = "who is abbas"
candidates = brain.search(query, k=20)

query_analysis = brain._analyze_query(query)

print(f"Query: '{query}'")
print(f"Type: {query_analysis['type']}")
print(f"Key terms: {query_analysis['key_terms']}")
print(f"Entities: {query_analysis['entities']}")
print()

print("ALL CANDIDATES WITH SCORES:")
print(f"{'Score':<8} {'File':<55} {'Type':<20}")
print("-" * 85)

scored = []
for doc in candidates:
    filename = doc.metadata.get('source_file', '')
    doc_info = brain.doc_summaries.get(filename, {})
    doc_type = doc_info.get('document_type', 'unknown')
    
    f_score = brain._score_filename_match(query_analysis, filename)
    t_score = brain._score_topic_match(query_analysis, doc_type, doc_info)
    c_score = brain._score_content_match(query_analysis, doc.page_content)
    e_score = brain._score_entity_match(query_analysis, doc_info)
    
    total = f_score + t_score + c_score + e_score
    
    scored.append((total, filename, doc_type, f_score, t_score, c_score, e_score))

scored.sort(key=lambda x: x[0], reverse=True)

for total, fname, dtype, fs, ts, cs, es in scored[:10]:
    print(f"{total:<8.0f} {fname:<55} {dtype:<20}")
    print(f"         F={fs:.0f}  T={ts:.0f}  C={cs:.0f}  E={es:.0f}")
    print()

# Now show what intelligent_search returns
print("\n" + "=" * 60)
print("WHAT intelligent_search() RETURNS:")
print("=" * 60)
results = brain.intelligent_search(query, k=4)
for i, doc in enumerate(results):
    print(f"  {i+1}. {doc.metadata.get('source_file')} (page {doc.metadata.get('page_number')})")