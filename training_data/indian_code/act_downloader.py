import requests
import time
import os
import re
from urllib.parse import quote, urljoin
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IndianActDownloader:
    def __init__(self, download_folder: str = "downloaded_acts"):
        self.base_url = "https://www.indiacode.nic.in"
        self.search_url = "https://www.indiacode.nic.in/handle/123456789/1362/simple-search"
        self.download_folder = download_folder
        self.session = requests.Session()
        
        # Create download folder if it doesn't exist
        os.makedirs(self.download_folder, exist_ok=True)
        
        # Set up headers to mimic a real browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
    
    def search_act(self, act_name: str) -> Optional[str]:
        """Step 1 & 2: Search for an act and get the view link from the second row"""
        try:
            # Prepare search query
            search_query = quote(act_name.replace(" ", " "))
            
            # Search parameters
            params = {
                'page-token': 'f9ab354f3ca9',
                'page-token-value': '44deac05525b1d89cf03689f6ba59bb8',
                'nccharset': 'E5306D80',
                'query': search_query,
                'btngo': '',
                'searchradio': 'acts'
            }
            
            logger.info(f"Searching for act: {act_name}")
            print(f"Search URL: {self.search_url}?{params}")
            response = self.session.get(self.search_url, params=params)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the table with search results
            table = soup.find('table', {'class': 'table table-hover'})
            if not table:
                logger.warning(f"No search results table found for: {act_name}")
                return None
            
            # Get tbody or use table directly
            tbody = table.find('tbody')
            if tbody:
                rows = tbody.find_all('tr')[1:]  # Skip header
            else:
                rows = table.find_all('tr')[1:]  # Skip header, no tbody
            
            if len(rows) < 1:
                logger.warning(f"No results found for: {act_name}")
                return None
            
            # Get the first relevant row (usually the most relevant result)
            target_row = None
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 4:
                    short_title = cells[2].get_text(strip=True)
                    # Check if this row contains our target act
                    if any(word.lower() in short_title.lower() for word in act_name.split() if len(word) > 3):
                        target_row = row
                        break
            
            if not target_row:
                target_row = rows[0]  # Fallback to first row
            
            # Extract the view link
            view_cell = target_row.find_all('td')[-1]  # Last cell contains the view link
            view_link = view_cell.find('a')
            
            if view_link and view_link.get('href'):
                full_link = urljoin(self.base_url, view_link['href'])
                logger.info(f"Found view link: {full_link}")
                return full_link
            else:
                logger.warning(f"No view link found for: {act_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error searching for act '{act_name}': {str(e)}")
            return None
    
    def get_pdf_download_link(self, view_url: str) -> Optional[str]:
        """Step 3: Get the PDF download link from the view page"""
        try:
            logger.info(f"Getting PDF link from: {view_url}")
            response = self.session.get(view_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for the PDF download link with the specific pattern
            # Pattern: <a style="font-weight: bold;font-size: 17px;margin: 0;" target="_blank" href="/bitstream/...pdf">
            pdf_links = soup.find_all('a', {'target': '_blank'})
            
            for link in pdf_links:
                href = link.get('href', '')
                # Check if it's a PDF link and contains the expected pattern
                if '.pdf' in href and '/bitstream/' in href:
                    # Check if it's English (not Hindi or other languages)
                    link_text = link.get_text(strip=True)
                    if link_text and not self._is_non_english_text(link_text):
                        full_pdf_url = urljoin(self.base_url, href)
                        logger.info(f"Found PDF download link: {full_pdf_url}")
                        return full_pdf_url
            
            # Fallback: look for any PDF link
            for link in soup.find_all('a', href=True):
                href = link['href']
                if '.pdf' in href and '/bitstream/' in href:
                    full_pdf_url = urljoin(self.base_url, href)
                    logger.info(f"Found PDF download link (fallback): {full_pdf_url}")
                    return full_pdf_url
            
            logger.warning("No PDF download link found")
            return None
            
        except Exception as e:
            logger.error(f"Error getting PDF link: {str(e)}")
            return None
    
    def _is_non_english_text(self, text: str) -> bool:
        """Check if text contains non-English characters (e.g., Hindi)"""
        # Simple check for Devanagari script (Hindi)
        hindi_range = range(0x0900, 0x097F)
        for char in text:
            if ord(char) in hindi_range:
                return True
        return False
    
    def download_pdf(self, pdf_url: str, act_name: str, year: str) -> Optional[str]:
        """Step 4: Download the PDF and save with proper name"""
        try:
            logger.info(f"Downloading PDF: {pdf_url}")
            response = self.session.get(pdf_url, stream=True)
            response.raise_for_status()
            
            # Create a clean filename
            clean_name = re.sub(r'[^\w\s-]', '', act_name)
            clean_name = re.sub(r'[-\s]+', '_', clean_name)
            filename = f"{clean_name}_{year}.pdf"
            filepath = os.path.join(self.download_folder, filename)
            
            # Download the file
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"Downloaded: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error downloading PDF: {str(e)}")
            return None
    
    def download_act(self, act_name: str, year: str) -> Optional[str]:
        """Complete process: search, find, and download an act"""
        logger.info(f"Starting download process for: {act_name} ({year})")
        
        # Step 1 & 2: Search and get view link
        view_url = self.search_act(act_name)
        if not view_url:
            return None
        
        # Add delay to be respectful to the server
        time.sleep(2)
        
        # Step 3: Get PDF download link
        pdf_url = self.get_pdf_download_link(view_url)
        if not pdf_url:
            return None
        
        # Add delay before downloading
        time.sleep(1)
        
        # Step 4: Download the PDF
        filepath = self.download_pdf(pdf_url, act_name, year)
        return filepath
    
    def download_acts_from_list(self, acts_list: List[Dict[str, str]]) -> Dict[str, str]:
        """Download multiple acts from a list"""
        results = {}
        
        for i, act_info in enumerate(acts_list, 1):
            act_name = act_info.get('name', '')
            year = act_info.get('year', '')
            
            if not act_name:
                continue
            
            logger.info(f"Processing act {i}/{len(acts_list)}: {act_name}")
            
            try:
                filepath = self.download_act(act_name, year)
                results[act_name] = filepath if filepath else "Failed"
                
                # Add delay between downloads to be respectful
                time.sleep(3)
                
            except Exception as e:
                logger.error(f"Failed to download {act_name}: {str(e)}")
                results[act_name] = "Error"
        
        return results

def main():
    # List of acts to download (from your table)
    acts_to_download = [
        {"name": "Aadhaar (Targeted Delivery of Financial and Other Subsidies) Act", "year": "2016"},
        {"name": "Births, Deaths and Marriages Registration Act", "year": "1886"},
        {"name": "Child and Adolescent Labour (Prohibition and Regulation) Act", "year": "1986"},
        {"name": "Code of Civil Procedure", "year": "1908"},
        {"name": "Dissolution of Muslim Marriages Act", "year": "1939"},
        {"name": "Dowry Prohibition Act", "year": "1961"},
        {"name": "Drugs and Cosmetics Act", "year": "1940"},
        {"name": "Employees' Provident Funds and Miscellaneous Provisions Act", "year": "1952"},
        {"name": "Hindu Marriage Act", "year": "1955"},
        {"name": "Indian Christian Marriage Act", "year": "1872"},
        {"name": "Indian Contract Act", "year": "1872"},
        {"name": "Juvenile Justice (Care and Protection of Children) Act", "year": "2016"},
        {"name": "Motor Vehicles Act", "year": "1988"},
        {"name": "Muslim Women (Protection of Rights on Marriage/Divorce) Acts", "year": "1986"},
        {"name": "Narcotic Drugs and Psychotropic Substances Act", "year": "1985"},
        {"name": "Payment of Gratuity Act", "year": "1972"},
        {"name": "Prohibition of Child Marriage Act", "year": "2007"},
        {"name": "Protection of Children from Sexual Offences (POCSO) Act", "year": "2012"},
        {"name": "Protection of Women from Domestic Violence Act", "year": "2005"},
        {"name": "Right of Children to Free and Compulsory Education Act", "year": "2009"},
        {"name": "Right to Information Act", "year": "2005"},
        {"name": "Special Marriage Act", "year": "1954"},
    ]
    
    # Create downloader instance
    downloader = IndianActDownloader()
    
    print("="*80)
    print("INDIAN ACTS DOWNLOADER")
    print("="*80)
    print(f"Download folder: {downloader.download_folder}")
    print(f"Total acts to download: {len(acts_to_download)}")
    print("="*80)
    
    # Download all acts
    results = downloader.download_acts_from_list(acts_to_download)
    
    # Print results
    print("\n" + "="*80)
    print("DOWNLOAD RESULTS")
    print("="*80)
    
    successful = 0
    failed = 0
    
    for act_name, result in results.items():
        status = "✓" if result and result != "Failed" and result != "Error" else "✗"
        print(f"{status} {act_name}")
        if result and result != "Failed" and result != "Error":
            print(f"   → {result}")
            successful += 1
        else:
            print(f"   → {result}")
            failed += 1
        print()
    
    print("="*80)
    print(f"Summary: {successful} successful, {failed} failed")
    print("="*80)

if __name__ == "__main__":
    main()
