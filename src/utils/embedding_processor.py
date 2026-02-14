# embedding 
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


class EmbeddingProcessor:
    
    def __init__(self, model_name, chunk_size=400, chunk_overlap=80):
        self.embedding_model = SentenceTransformer(model_name, trust_remote_code=True)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def compute_paper_scores(self, papers, user_query, weights):

        query_embedding = self.embedding_model.encode(user_query, normalize_embeddings=True)
        
        paper_scores = []
        
        print("\nComputing weighted scores...")
        
        α = weights['alpha']
        β = weights['beta']
        γ = weights['gamma']
        δ = weights['delta']
        τ = weights['tau']
        λ = weights['lambda']
        
        for paper in papers:
            chunks = self.text_splitter.split_text(paper['summary'])
            
            if not chunks:
                continue
            
            chunk_embeddings = self.embedding_model.encode(chunks, normalize_embeddings=True)
            similarities = np.dot(chunk_embeddings, query_embedding)
            
            max_similarity = float(np.max(similarities))
            freshness = np.exp(-λ * paper['days_ago'])
            
            relevant_sims = similarities[similarities > τ]
            coverage_avg = float(np.mean(relevant_sims)) if len(relevant_sims) > 0 else 0.0
            
            n_relevant = int(np.sum(similarities > τ))
            n_total = len(similarities)
            volume_bonus = np.log(1 + n_relevant) / np.log(1 + n_total)
            
            final_score = (
                α * max_similarity +
                β * freshness +
                γ * coverage_avg +
                δ * volume_bonus
            )
            
            paper_scores.append({
                'paper': paper,
                'score': final_score,
                'max_similarity': max_similarity,
                'coverage_avg': coverage_avg,
                'volume_bonus': volume_bonus,
                'freshness': freshness,
                'n_relevant': n_relevant,
                'chunks': chunks,
                'chunk_similarities': similarities
            })
        
        paper_scores.sort(key=lambda x: x['score'], reverse=True)
        return paper_scores
    
    def extract_top_chunks(self, pdf_papers, user_query, top_k=10):

        query_embedding = self.embedding_model.encode(user_query, normalize_embeddings=True)
        all_chunks = []
        
        print("\nExtracting top chunks from PDFs...")
        
        for pdf_data in pdf_papers:
            chunks = self.text_splitter.split_text(pdf_data['full_text'])
            chunk_embeddings = self.embedding_model.encode(chunks, normalize_embeddings=True)
            similarities = np.dot(chunk_embeddings, query_embedding)
            
            for chunk, sim in zip(chunks, similarities):
                all_chunks.append({
                    'text': chunk,
                    'similarity': float(sim),
                    'paper_title': pdf_data['paper']['title'],
                    'paper_id': pdf_data['paper']['id']
                })
        
        all_chunks.sort(key=lambda x: x['similarity'], reverse=True)
        top_chunks = all_chunks[:top_k]
        
        print(f"Selected {len(top_chunks)} most relevant chunks")
        return top_chunks
    
    def compute_ranking_loss(self, user_query, papers, paper_scores, k=10):

        print(f"\nComputing DCG-based Ranking Loss...")
        
        query_emb = self.embedding_model.encode(user_query, normalize_embeddings=True)
        
        ground_truth_relevances = []
        for paper in papers:
            paper_text = f"{paper['title']} {paper['summary']}"
            paper_emb = self.embedding_model.encode(paper_text, normalize_embeddings=True)
            rel = float(np.dot(query_emb, paper_emb))
            ground_truth_relevances.append(rel)
        
        sorted_gt_rels = sorted(ground_truth_relevances, reverse=True)[:k]
        
        idcg = 0.0
        for i, rel in enumerate(sorted_gt_rels):
            idcg += (2**rel - 1) / np.log2(i + 2)
        
        top_k_indices = [papers.index(ps['paper']) for ps in paper_scores[:k]]
        
        dcg = 0.0
        for rank, paper_idx in enumerate(top_k_indices):
            rel = ground_truth_relevances[paper_idx]
            dcg += (2**rel - 1) / np.log2(rank + 2)
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        loss = 1.0 - ndcg
        
        top_k_rels = [ground_truth_relevances[idx] for idx in top_k_indices]
        precision_at_k = sum(1 for rel in top_k_rels if rel > 0.5) / k
        
        metrics = {
            'ndcg@k': ndcg,
            'dcg': dcg,
            'idcg': idcg,
            'loss': loss,
            'precision@k': precision_at_k,
            'mean_rel_top_k': np.mean(top_k_rels),
            'mean_rel_all': np.mean(ground_truth_relevances),
            'improvement': np.mean(top_k_rels) - np.mean(ground_truth_relevances)
        }
        
        print(f"   NDCG@{k}: {ndcg:.3f}")
        print(f"   Loss: {loss:.3f}")
        print(f"   Precision@{k}: {precision_at_k:.3f}")
        
        return loss, metrics

