# ingest_all.py
"""ONE-TIME SCRIPT: Ingest all PDFs into the knowledge base"""
import os
import sys
import time
from brain import KnowledgeBrain

def main():
    print("=" * 60)
    print("🧠 BUILDING THE KNOWLEDGE BRAIN")
    print("=" * 60)
    
    pdf_dir = "./pdfs"
    vector_dir = "./vector_store"
    
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
        print(f"\n✅ Created: {pdf_dir}")
        print("   Add your PDFs there and run again.")
        sys.exit(0)
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"\n❌ No PDFs in: {pdf_dir}")
        print("   Add your PDFs there and run again.")
        sys.exit(1)
    
    print(f"\n📚 Found {len(pdf_files)} PDFs\n")
    
    start = time.time()
    
    brain = KnowledgeBrain(
        pdf_directory=pdf_dir,
        persist_directory=vector_dir
    )
    
    results = brain.ingest_all_pdfs()
    
    elapsed = time.time() - start
    
    print("\n" + "=" * 60)
    print("✅ COMPLETE!")
    print("=" * 60)
    print(f"\nProcessed: {results['processed']}")
    print(f"Skipped: {results['skipped']}")
    print(f"Failed: {results['failed']}")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    
    stats = brain.get_stats()
    print(f"\nBrain Stats:")
    print(f"  Documents: {stats['total_documents']}")
    print(f"  Chunks: {stats['total_chunks']:,}")
    print(f"  Pages: {stats['total_pages']:,}")
    print(f"  Size: {stats['brain_size_mb']} MB")
    
    print(f"\n🚀 Run 'python app.py' to start!")

if __name__ == "__main__":
    main()