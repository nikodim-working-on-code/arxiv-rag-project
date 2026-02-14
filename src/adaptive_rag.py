# main class - please remember that "promts" is smth u have to adapt for urself
import arxiv
import fitz
import json
import os
import shutil
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from datetime import datetime
from groq import Groq
from src.config import Config
from src.utils.arxiv_fetcher import ArXivFetcher
from src.utils.embedding_processor import EmbeddingProcessor
from src.utils.optimizer import WeightOptimizer


class AdaptiveRAG:    
    def __init__(self, weights_file=None, history_file=None):
        self.weights_file = weights_file or Config.WEIGHTS_FILE
        self.history_file = history_file or Config.HISTORY_FILE
        
        self.fetcher = ArXivFetcher()
        self.processor = EmbeddingProcessor(
            Config.EMBEDDING_MODEL,
            Config.CHUNK_SIZE,
            Config.CHUNK_OVERLAP
        )
        self.optimizer = WeightOptimizer()
        
        self.groq_client = Groq(api_key=Config.GROQ_API_KEY)
        self.llm_model = Config.GROQ_MODEL
        
        # weights
        self.weights = self.load_weights()
        
        print(f"\n Loaded weights:")
        print(f"   α (max_similarity): {self.weights['alpha']:.3f}")
        print(f"   β (freshness): {self.weights['beta']:.3f}")
        print(f"   γ (coverage_avg): {self.weights['gamma']:.3f}")
        print(f"   δ (volume_bonus): {self.weights['delta']:.3f}")
        print(f"   τ (threshold): {self.weights['tau']:.3f}")
        print(f"   λ (freshness_decay): {self.weights['lambda']:.3f}")
        print(f"   Total queries processed: {self.weights['num_queries']}\n")
    
    def load_weights(self):
        if os.path.exists(self.weights_file):
            try:
                with open(self.weights_file, 'r') as f:
                    weights = json.load(f)
                print(f" Loaded weights from {self.weights_file}")
                return weights
            except Exception as e:
                print(f" Error loading weights: {e}. Using defaults.")
        
        return Config.DEFAULT_WEIGHTS.copy()
    
    def save_weights(self):
        try:
            with open(self.weights_file, 'w') as f:
                json.dump(self.weights, f, indent=4)
            print(f" Weights saved to {self.weights_file}")
        except Exception as e:
            print(f" Error saving weights: {e}")
    
    def log_optimization_step(self, user_query, loss, metrics):
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                history = json.load(f)
        else:
            history = []
        
        history.append({
            'timestamp': datetime.now().isoformat(),
            'query': user_query,
            'weights': self.weights.copy(),
            'loss': loss,
            'metrics': metrics
        })
        
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=4)
    
    def _call_groq(self, prompt, temperature=0.3):
        try:
            response = self.groq_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that outputs valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f" Groq API error: {e}")
            return None
    
    def parse_query(self, user_query):
        analysis_prompt = f"""You MUST output ONLY valid JSON. No explanations, no markdown.

Query: "{user_query}"

Extract these fields:
1. categories (list): Pick from cs.AI, cs.LG, cs.RO, cs.CV, cs.CL
   - "machine learning" OR "ML" → ["cs.LG"]
   - "robotics" → ["cs.RO"]
   - "computer vision" → ["cs.CV"]
   - "NLP" OR "language" → ["cs.CL"]
   - "AI" (general) → ["cs.AI"]
   - Default: ["cs.LG"]

2. time_range_days (int):
   - "today" → 1
   - "recent" OR "week" → 7
   - "2 weeks" OR "14 days" → 14
   - "month" → 30
   - Default: 7

3. max_papers (int): Always 20

Examples:
{{"categories": ["cs.LG"], "time_range_days": 14, "max_papers": 20}}
{{"categories": ["cs.RO"], "time_range_days": 7, "max_papers": 20}}

Output ONLY JSON:"""
        
        response = self._call_groq(analysis_prompt, temperature=0.1)
        
        try:
            if response and response.startswith('```'):
                response = response.split('```')[1]
                if response.startswith('json'):
                    response = response[4:]
            
            params = json.loads(response.strip())
            
            if 'categories' not in params or not params['categories']:
                params['categories'] = ['cs.LG']
            if 'time_range_days' not in params:
                params['time_range_days'] = 7
            if 'max_papers' not in params:
                params['max_papers'] = 20
        except:
            print(f" Failed to parse JSON, using keyword fallback")
            params = self._fallback_parse(user_query)
        
        print(f'FINAL PARAMS: {params}')
        return params
    
    def _fallback_parse(self, user_query):
        query_lower = user_query.lower()
        
        categories = []
        if 'machine learning' in query_lower or ' ml ' in query_lower:
            categories.append('cs.LG')
        if 'robot' in query_lower:
            categories.append('cs.RO')
        if 'vision' in query_lower or 'image' in query_lower:
            categories.append('cs.CV')
        if 'nlp' in query_lower or 'language' in query_lower:
            categories.append('cs.CL')
        if 'ai' in query_lower and not categories:
            categories.append('cs.AI')
        if not categories:
            categories = ['cs.LG']
        
        if 'today' in query_lower:
            time_range = 1
        elif '2 week' in query_lower or '14 day' in query_lower:
            time_range = 14
        elif 'week' in query_lower or 'recent' in query_lower:
            time_range = 7
        elif 'month' in query_lower:
            time_range = 30
        else:
            time_range = 7
        
        return {
            "categories": categories,
            "time_range_days": time_range,
            "max_papers": 20
        }
    
    def fetch_top_pdfs(self, top_papers):
        os.makedirs(Config.TEMP_PDF_DIR, exist_ok=True)
        pdf_papers = []
        
        print("\n Downloading top-3 papers as PDFs...")
        
        for i, paper_data in enumerate(top_papers[:3]):
            paper = paper_data['paper']
            
            try:
                search = arxiv.Search(id_list=[paper['id']])
                result = next(arxiv.Client().results(search))
                
                pdf_path = f"{Config.TEMP_PDF_DIR}/{paper['id']}.pdf"
                result.download_pdf(filename=pdf_path)
                
                print(f"  Downloaded: {paper['title'][:60]}...")
                
                try:
                    doc = fitz.open(pdf_path)
                    full_text = ""
                    
                    for page_num, page in enumerate(doc):
                        text = page.get_text()
                        
                        if 'References' in text or 'REFERENCES' in text:
                            ref_index = text.find('References')
                            if ref_index == -1:
                                ref_index = text.find('REFERENCES')
                            
                            if ref_index > 0:
                                text = text[:ref_index]
                                full_text += f"\n--- Page {page_num + 1} ---\n{text}"
                                print(f"   Stopped at References (page {page_num + 1})")
                                break
                        
                        full_text += f"\n--- Page {page_num + 1} ---\n{text}"
                    
                    doc.close()
                    print(f" Extracted {len(full_text)} characters from PDF")
                except Exception as e:
                    print(f"   Failed to parse PDF: {e}")
                    full_text = f"Title: {paper['title']}\n\nAbstract: {paper['summary']}"
                
                pdf_papers.append({
                    'paper': paper,
                    'full_text': full_text,
                    'pdf_path': pdf_path
                })
            except Exception as e:
                print(f"  Error downloading {paper['id']}: {e}")
        
        return pdf_papers
    
    def generate_answer(self, user_query, top_chunks):
        unique_papers = {}
        for chunk in top_chunks:
            paper_id = chunk['paper_id']
            if paper_id not in unique_papers:
                unique_papers[paper_id] = {
                    'title': chunk['paper_title'],
                    'paper_id': paper_id,
                    'chunks': []
                }
            unique_papers[paper_id]['chunks'].append(chunk['text'])
        
        context_with_sources = []
        for i, (paper_id, data) in enumerate(unique_papers.items(), 1):
            combined_text = "\n\n".join(data['chunks'])
            source_label = f"[Source {i}: {data['title'][:50]}...]"
            context_with_sources.append(f"{source_label}\n{combined_text}")
        
        context = "\n\n".join(context_with_sources)
        
        prompt = f"""You are a research assistant. Answer the question using the provided context from recent arXiv papers.

IMPORTANT RULES:
1. DO NOT copy mathematical formulas or equations from the context
2. Explain concepts in plain language
3. Focus on high-level methods and findings
4. Cite sources as (Source X)

TASK: The user asked "{user_query}". Based on the recent papers, summarize:
1. What mathematical techniques or methods were proposed
2. What problems are being addressed
3. Key theoretical contributions

Context:
{context}

Question: {user_query}

Answer (explain in plain language, no formulas):"""
        
        print("\n Generating answer...\n")
        answer = self._call_groq(prompt, temperature=0.5)
        
        if not answer:
            answer = " Failed to generate answer (API error)"
        
        sources_list = "\n\n" + "="*70 + "\nSources:\n"
        for i, (paper_id, data) in enumerate(unique_papers.items(), 1):
            sources_list += f"{i}. {data['title']} (arXiv:{paper_id})\n"
        
        return answer + sources_list
    
    def run_full_pipeline(self, user_query):
        print("="*70)
        print(f" User Query: {user_query}")
        print("="*70)
        
        # step 1 parse query
        print("\n Step 1: Analyzing query...")
        params = self.parse_query(user_query)
        print(f"   Categories: {params['categories']}")
        print(f"   Time range: {params['time_range_days']} days")
        
        # step 2 fetch papers
        print("\n Step 2: Fetching papers...")
        papers = self.fetcher.fetch_papers_per_day(params)
        print(f"\n    Total fetched: {len(papers)} papers")
        
        if not papers:
            print(" No papers found!")
            return None
        
        # step 3 rank papers
        print("\n Step 3: Chunking and scoring...")
        paper_scores = self.processor.compute_paper_scores(papers, user_query, self.weights)
        
        print(f"\n    Top 5 papers by weighted score:")
        for i, ps in enumerate(paper_scores[:5], 1):
            print(f"   {i}. [{ps['score']:.3f}] {ps['paper']['title'][:60]}...")
            print(f"      max_sim={ps['max_similarity']:.2f}, fresh={ps['freshness']:.2f}, "
                  f"cov_avg={ps['coverage_avg']:.2f}, vol={ps['volume_bonus']:.2f}")
        
        # step 4 compute loss
        loss, metrics = self.processor.compute_ranking_loss(user_query, papers, paper_scores, 
k=10)
        
        # step 5 update weights
        self.weights = self.optimizer.update_adaptive(self.weights, loss, metrics)
        self.save_weights()
        
        # step 6 log optimization
        self.log_optimization_step(user_query, loss, metrics)
        
        # step 7 fetching PDFs + answer
        pdf_papers = self.fetch_top_pdfs(paper_scores)
        top_chunks = self.processor.extract_top_chunks(pdf_papers, user_query)
        
        print("\n" + "="*70)
        print(" FINAL ANSWER")
        print("="*70)
        answer = self.generate_answer(user_query, top_chunks)
        print(answer)
        print("="*70)
        
        if os.path.exists(Config.TEMP_PDF_DIR):
            shutil.rmtree(Config.TEMP_PDF_DIR)
        
        return {
            'query': user_query,
            'answer': answer,
            'loss': loss,
            'metrics': metrics,
            'top_papers': paper_scores[:5]
        }

