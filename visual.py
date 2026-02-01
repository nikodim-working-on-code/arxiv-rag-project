import os

import arxiv
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE


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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50, separators=["\n\n", "\n", ". ", " "])
    
    papers_sums = fetch_papers()
    text = papers_sums[0]['summary']
    chunks = text_splitter.split_text(text)
    chunk_embeddings = embedding_model.encode(chunks, normalize_embeddings=True, show_progress_bar=True)
    
    chroma_client = chromadb.Client(Settings(anonymized_telemetry=False, allow_reset=True))
    try:
        chroma_client.delete_collection("arxiv_chunks")
    except:
        pass
    collection = chroma_client.create_collection(name="arxiv_chunks", metadata={"description": "Chunks from arXiv AI papers"})
    collection.add(
        ids=[f"{papers_sums[0]['id']}_chunk_{i}" for i in range(len(chunks))],
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

    # Получаем данные
    result = collection.get(include=['embeddings', 'documents'])
    embeddings = np.array(result['embeddings'])
    documents = result['documents']
    
    # Вычисляем матрицу схожести
    similarity_matrix = cosine_similarity(embeddings)
    
    # Настройки для графа
    threshold = 0.7  # Порог схожести
    
    # t-SNE в 3D для вращения
    tsne = TSNE(n_components=3, random_state=42, perplexity=min(5, len(embeddings)-1))
    coords = tsne.fit_transform(embeddings)
    
    # Создаём 3D граф
    fig = go.Figure()
    
    # Добавляем рёбра с подписями
    edge_x = []
    edge_y = []
    edge_z = []
    edge_labels_x = []
    edge_labels_y = []
    edge_labels_z = []
    edge_labels_text = []
    
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            if similarity_matrix[i][j] > threshold:
                # Линия между узлами
                edge_x.extend([coords[i, 0], coords[j, 0], None])
                edge_y.extend([coords[i, 1], coords[j, 1], None])
                edge_z.extend([coords[i, 2], coords[j, 2], None])
                
                # Позиция текста (середина ребра)
                mid_x = (coords[i, 0] + coords[j, 0]) / 2
                mid_y = (coords[i, 1] + coords[j, 1]) / 2
                mid_z = (coords[i, 2] + coords[j, 2]) / 2
                
                edge_labels_x.append(mid_x)
                edge_labels_y.append(mid_y)
                edge_labels_z.append(mid_z)
                edge_labels_text.append(f'{similarity_matrix[i][j]:.2f}')
    
    # Рёбра
    fig.add_trace(go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode='lines',
        line=dict(color='rgba(69,69,69,0.5)', width=2),
        hoverinfo='none',
        showlegend=False
    ))
    
    # Подписи на рёбрах (сила связи)
    fig.add_trace(go.Scatter3d(
        x=edge_labels_x,
        y=edge_labels_y,
        z=edge_labels_z,
        mode='text',
        text=edge_labels_text,
        textfont=dict(size=10, color='red'),
        hoverinfo='text',
        hovertext=[f'Similarity: {t}' for t in edge_labels_text],
        showlegend=False
    ))
    
    # Узлы (чанки)
    fig.add_trace(go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode='markers+text',
        text=[f'C{i}' for i in range(len(embeddings))],
        textposition="top center",
        marker=dict(
            size=15,
            color=np.arange(len(embeddings)),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Chunk ID", x=1.1),
            line=dict(width=2, color='black')
        ),
        hovertext=[f"<b>Chunk {i}</b><br>{doc[:100]}..." for i, doc in enumerate(documents)],
        hoverinfo='text',
        showlegend=False
    ))
    
    fig.update_layout(
        title=f'3D граф связей между чанками (similarity > {threshold})',
        width=1200,
        height=900,
        scene=dict(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            zaxis_title='Dimension 3',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        hovermode='closest'
    )
    
    fig.write_html('chunk_graph_3d.html')
    fig.show()
    
    print(f"\n✅ 3D граф сохранён в chunk_graph_3d.html")
    print(f"📊 Порог схожести: {threshold}")
    print(f"🔗 Количество связей: {np.sum(similarity_matrix > threshold) // 2}")
    
    # Матрица схожести
    print("\n📈 Матрица схожести:")
    print(f"{'':10}", end='')
    for i in range(len(embeddings)):
        print(f"  C{i}", end='')
    print()
    for i in range(len(embeddings)):
        print(f"Chunk {i:2d}  ", end='')
        for j in range(len(embeddings)):
            if i == j:
                print(" 1.00", end='')
            else:
                color = '\033[92m' if similarity_matrix[i][j] > threshold else '\033[0m'
                print(f"{color}{similarity_matrix[i][j]:5.2f}\033[0m", end='')
        print()
