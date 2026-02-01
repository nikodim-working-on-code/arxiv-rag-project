import arxiv
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings
import json



def fetch_papers(max_results=3):
    """Получаем статьи с arXiv"""
    client = arxiv.Client()
    search = arxiv.Search(
        query="cat:cs.AI",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    papers = []
    for result in client.results(search):
        papers.append({
            'id': result.entry_id.split('/')[-1].split('v')[0],
            'title': result.title,
            'summary': result.summary,
            'authors': ', '.join([a.name for a in result.authors[:3]]),
            'published': result.published.strftime('%Y-%m-%d')
        })
    
    return papers


if __name__ == "__main__":
    
    embedding_model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap = 50, separators=["\n\n", "\n", ". ", " "])
    
    papers_sums = fetch_papers()
    text = papers_sums[0]['summary']
    chunks = text_splitter.split_text(text)
    chunk_embeddings = embedding_model.encode(chunks, normalize_embeddings=True, show_progress_bar=True)
    
    chroma_client = chromadb.Client(Settings(anonymized_telemetry = False, allow_reset= True))
    try:
        chroma_client.delete_collection("arxiv_chunks")
    except:
        pass
    collection = chroma_client.create_collection(name="arxiv_chunks", metadata={"description": "Chunks from arXiv AI papers"})
    collection.add(ids=[f"{papers_sums[0]['id']}_chunk_{i}" for i in range(len(chunks))],
                   embeddings=chunk_embeddings.tolist(),
                   documents=chunks,
                    metadatas=[
                        {
                            'paper_id': papers_sums[0]['id'],
                            'title': papers_sums[0]['title'],
                            'authors': papers_sums[0]['authors'],
                            'chunk_index': i
                        }
                        for i in range(len(chunks))
                    ]
                )

    # print(json.dumps(collection.get(), indent=2, ensure_ascii=False))
    
    user_query = "How much training data was collected?"

    query_embedding = embedding_model.encode(
        user_query,
        normalize_embeddings=True
    )
    
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3
    )
    
    for i, (chunk_id, distance, metadata, document) in enumerate(
        zip(
            results['ids'][0],
            results['distances'][0],
            results['metadatas'][0],
            results['documents'][0]
        ),
        start=1
    ):
        
        # Chroma возвращает L2 distance, конвертируем в cosine similarity
        # Формула: similarity ≈ 1 - (distance² / 2) для нормализованных векторов
        similarity = 1 - (distance ** 2 / 2)
        
        print(f"📄 Результат #{i}")
        print(f"   Similarity: {similarity:.4f} | Distance: {distance:.4f}")
        print(f"   Paper: {metadata['title'][:60]}...")
        print(f"   Chunk #{metadata['chunk_index']}")
        print(f"   Text: {document[:150]}...")
        print()
    