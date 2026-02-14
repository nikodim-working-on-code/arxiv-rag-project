# paper fetching
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
import time


class ArXivFetcher:
    
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
    
    def fetch_papers_per_day(self, params):
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=params['time_range_days'])
        
        print(f"Target: {params['time_range_days']} days from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Goal: {params['max_papers']} papers per day\n")
        
        all_papers = []
        
        for category in params['categories']:
            print(f"Category: {category}")
            category_papers = self._fetch_category(category, start_date, end_date)
            
            filtered = [p for p in category_papers if start_date <= p['published'] <= end_date]
            print(f"Selected {len(filtered)} papers in target range\n")
            all_papers.extend(filtered)
        
        print(f"Total papers: {len(all_papers)}\n")
        
        if not all_papers:
            return []
        
        return self._select_top_per_day(all_papers, params['max_papers'])
    
    def _fetch_category(self, category, start_date, end_date):
        papers = []
        page_size = 100
        start_index = 0
        iteration = 1
        max_iterations = 50
        
        while iteration <= max_iterations:
            params = {
                'search_query': f'cat:{category}',
                'start': start_index,
                'max_results': page_size,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            
            print(f"Iteration {iteration}: fetching {page_size} papers (start={start_index})...")
            
            try:
                response = requests.get(self.base_url, params=params, timeout=30)
                
                if response.status_code != 200:
                    print(f"HTTP {response.status_code}")
                    break
                
                entries = self._parse_response(response.content)
                
                if not entries:
                    print(f"No more results")
                    break
                
                papers.extend(entries)
                print(f"Fetched {len(entries)} papers (total: {len(papers)})")
                
                if papers:
                    oldest = min(papers, key=lambda x: x['published'])
                    days_covered = (end_date - oldest['published']).days
                    print(f"Oldest: {oldest['published'].strftime('%Y-%m-%d')} ({days_covered} days)")
                    
                    if oldest['published'] <= start_date:
                        print(f"Covered all days!\n")
                        break
                
                start_index += page_size
                iteration += 1
                time.sleep(3)
                
            except Exception as e:
                print(f"Error: {e}")
                break
        
        return papers
    
    def _parse_response(self, content):
        root = ET.fromstring(content)
        ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
        
        entries = []
        for entry in root.findall('atom:entry', ns):
            entry_id = entry.find('atom:id', ns).text
            title = entry.find('atom:title', ns).text.replace('\n', ' ').strip()
            summary = entry.find('atom:summary', ns).text.replace('\n', ' ').strip()
            published = entry.find('atom:published', ns).text
            
            pub_date = datetime.strptime(published, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
            
            authors = entry.findall('atom:author', ns)
            author_names = [a.find('atom:name', ns).text for a in authors[:3]]
            
            pdf_link = entry.find("atom:link[@title='pdf']", ns)
            pdf_url = pdf_link.get('href') if pdf_link is not None else ''
            
            entries.append({
                'id': entry_id.split('/')[-1].split('v')[0],
                'title': title,
                'summary': summary,
                'authors': ', '.join(author_names),
                'published': pub_date,
                'days_ago': (datetime.now(timezone.utc) - pub_date).days,
                'pdf_url': pdf_url
            })
        
        return entries
    
    def _select_top_per_day(self, papers, max_papers):
        papers_by_day = {}
        for paper in papers:
            day_key = paper['published'].strftime('%Y-%m-%d')
            if day_key not in papers_by_day:
                papers_by_day[day_key] = []
            papers_by_day[day_key].append(paper)
        
        print(f"Distribution (taking {max_papers} newest per day):\n")
        
        selected = []
        for day in sorted(papers_by_day.keys(), reverse=True):
            day_papers = papers_by_day[day]
            day_papers.sort(key=lambda x: x['published'], reverse=True)
            top = day_papers[:max_papers]
            selected.extend(top)
            print(f"{day}: {len(top):2d}/{len(day_papers):2d} papers")
        
        print(f"\nTotal selected: {len(selected)} papers across {len(papers_by_day)} days")
        return selected

