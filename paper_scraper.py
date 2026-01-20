import requests
from bs4 import BeautifulSoup
import csv
import argparse
import sys
import os
import re
from urllib.parse import urljoin, unquote
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from wordcloud import WordCloud, STOPWORDS

'''
method:
python paper_scraper.py -c {conference} -y {year} -m {method} -w {workers}
'''

def generate_wordcloud(papers, conference, year):
    """
    Generate and save a word cloud from paper abstracts.
    """
    print("Generating word cloud...")
    
    # Aggregate all abstracts
    text = " ".join([p.get('abstract', '') for p in papers if p.get('abstract')])
    
    if not text:
        print("No abstract text available for word cloud.")
        return

    # Add custom stopwords
    stopwords = set(STOPWORDS)
    custom_stopwords = {
        # General academic terms
        'paper', 'method', 'methods', 'methodology', 'propose', 'proposed', 'approach', 'model', 'framework',
        'pipeline', 'network', 'algorithm', 'architecture', 'module', 'component', 'system', 
        'scheme', 'strategy', 'mechanism', 'baseline', 'benchmark', 'dataset', 'across',
        'data', 'database', 'corpus', 'collection', 'experiment', 'experiments', 'experimental', 
        'study', 'work', 'research', 'project', 'investigation', 'analysis', 'evaluation', 
        'ablation', 'comparison', 'discussion', 'introduction', 'conclusion', 'result', 
        'results', 'outcome', 'finding', 'observation', 'evidence', 'performance', 'accuracy', 
        'precision', 'recall', 'score', 'metric', 'error', 'loss', 'cost', 'objective', 
        'task', 'problem', 'challenge', 'issue', 'difficulty', 'limitation', 'drawback', 
        'gap', 'application', 'usage', 'utility', 'capability', 'ability', 'capacity', 
        'potential', 'quality', 'quantity', 'effectiveness', 'efficiency', 'robustness', 
        'complexity', 'state-of-the-art', 'state', 'art', 'sota', 'cutting-edge', 'advanced', 'novel', 'new',
        'innovative', 'modern', 'traditional', 'conventional', 'classical', 'existing', 
        'previous', 'prior', 'past', 'current', 'present', 'future', 'recent', 'related',
        
        # Verbs (actions)
        'use', 'using', 'used', 'utilize', 'utilizing', 'utilized', 'leverage', 'leveraging', 
        'employ', 'employing', 'adopt', 'adopting', 'adapt', 'adapting', 'apply', 'applying', 
        'make', 'making', 'generate', 'generating', 'produce', 'producing', 'create', 'creating', 
        'build', 'building', 'construct', 'constructing', 'design', 'designing', 'develop', 
        'developing', 'implement', 'implementing', 'train', 'training', 'trained', 'learn', 
        'learning', 'learned', 'study', 'studying', 'investigate', 'investigating', 'explore', 
        'exploring', 'examine', 'examining', 'analyze', 'analyzing', 'consider', 'considering', 
        'incorporate', 'incorporating', 'integrate', 'integrating', 'combine', 'combining', 
        'fuse', 'fusing', 'enable', 'enabling', 'allow', 'allowing', 'facilitate', 'facilitating', 
        'support', 'supporting', 'assist', 'assisting', 'help', 'helping', 'guide', 'guiding', 
        'require', 'requiring', 'need', 'needing', 'demand', 'demanding', 'provide', 'providing', 
        'offer', 'offering', 'present', 'presenting', 'introduce', 'introducing', 'suggest', 
        'suggesting', 'describe', 'describing', 'discuss', 'discussing', 'demonstrate', 
        'demonstrating', 'show', 'showing', 'exhibit', 'exhibiting', 'display', 'displaying', 
        'illustrate', 'illustrating', 'indicate', 'indicating', 'reveal', 'revealing', 
        'validate', 'validating', 'verify', 'verifying', 'evaluate', 'evaluating', 'assess', 
        'assessing', 'measure', 'measuring', 'test', 'testing', 'compare', 'comparing', 
        'contrast', 'contrasting', 'outperform', 'outperforming', 'surpass', 'surpassing', 
        'exceed', 'exceeding', 'beat', 'beating', 'improve', 'improving', 'enhance', 
        'enhancing', 'boost', 'boosting', 'increase', 'increasing', 'decrease', 'decreasing', 
        'reduce', 'reducing', 'minimize', 'minimizing', 'maximize', 'maximizing', 'optimize', 
        'optimizing', 'refine', 'refining', 'adjust', 'adjusting', 'address', 'addressing', 
        'tackle', 'tackling', 'solve', 'solving', 'handle', 'handling', 'deal', 'dealing', 
        'cope', 'coping', 'aim', 'aiming', 'focus', 'focusing', 'target', 'targeting', 
        'achieve', 'achieving',
        
        # Adjectives/Adverbs (descriptions)
        'extensive', 'comprehensive', 'thorough', 'systematic', 'detailed', 'in-depth', 
        'brief', 'short', 'long', 'large', 'small', 'high', 'low', 'good', 'bad', 'better', 
        'best', 'worse', 'worst', 'superior', 'inferior', 'significant', 'substantial', 
        'marginal', 'remarkable', 'notable', 'dramatic', 'slight', 'successful', 'effective', 
        'efficient', 'accurate', 'precise', 'robust', 'stable', 'reliable', 'flexible', 
        'scalable', 'general', 'generic', 'universal', 'specific', 'particular', 'distinct', 
        'different', 'various', 'diverse', 'multiple', 'single', 'simple', 'complex', 
        'complicated', 'difficult', 'hard', 'challenging', 'easy', 'fast', 'slow', 'rapid', 
        'quick', 'real-time', 'online', 'offline', 'static', 'dynamic', 'automatic', 
        'automated', 'manual', 'supervised', 'unsupervised', 'self-supervised', 
        'semi-supervised', 'weakly-supervised', 'deep', 'neural', 'visual', 'vision', 
        'specifically', 'additionally', 'moreover', 'furthermore', 'however', 'nonetheless', 
        'nevertheless', 'therefore', 'thus', 'hence', 'meanwhile', 'subsequently', 
        'consequently', 'accordingly', 'typically', 'usually', 'often', 'generally', 
        'commonly', 'widely', 'recently', 'currently', 'finally', 'eventually', 'besides', 
        'successfully', 'effectively', 'efficiently', 'accurately', 'robustly', 
        'significantly', 'greatly', 'highly', 'widely', 'mainly', 'mostly', 'largely', 
        'partially', 'fully', 'completely', 'totally', 'entirely', 'merely', 'simply', 
        'just', 'only', 'also', 'even', 'still', 'yet', 'already', 'via', 'based', 'using',
        
        # Nouns (general entities)
        'input', 'output', 'feature', 'features', 'representation', 'representations', 
        'structure', 'structures', 'context', 'contexts', 'level', 'levels', 'stage', 
        'stages', 'step', 'steps', 'process', 'processes', 'procedure', 'procedures', 
        'operation', 'operations', 'action', 'actions', 'activity', 'activities', 
        'behavior', 'behaviors', 'property', 'properties', 'characteristic', 
        'characteristics', 'attribute', 'attributes', 'aspect', 'aspects', 'factor', 
        'factors', 'element', 'elements', 'part', 'parts', 'detail', 'details', 
        'environment', 'environments', 'setting', 'settings', 'scenario', 'scenarios', 
        'case', 'cases', 'example', 'examples', 'instance', 'instances', 'sample', 
        'samples', 'number', 'numbers', 'value', 'values', 'rate', 'rates', 'ratio', 
        'ratios', 'code', 'available', 'publicly', 'github', 'https', 'url', 'link', 
        'page', 'web', 'site', 'term', 'terms', 'end-to-end', 'two', 'three', 'one', 
        'first', 'second', 'third'
    }
    
    # Use regex to filter out custom stopwords and their plural forms (s/es)
    # This aligns with the strategy used in keyword classification
    print("Filtering custom stopwords using regex (including plurals)...")
    sorted_stopwords = sorted(list(custom_stopwords), key=len, reverse=True)
    
    for word in sorted_stopwords:
        # Pattern: boundary + word + optional s/es + boundary
        # We use a negative lookbehind/lookahead or just match non-word chars?
        # The user's pattern: r'(?:^|[^a-zA-Z0-9])' + re.escape(keyword) + r'(?:s|es)?(?:$|[^a-zA-Z0-9])'
        # This matches the surrounding delimiters too. 
        # If we replace with space, we might consume punctuation we want to keep?
        # Actually for word cloud, punctuation is mostly ignored or removed.
        # But we don't want to merge "end." and "Start" into "endStart" if we consume the dot.
        # Replacing with " " is safe.
        pattern = r'(?:^|[^a-zA-Z0-9])' + re.escape(word) + r'(?:s|es)?(?:$|[^a-zA-Z0-9])'
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)

    # Generate word cloud
    # Note: We still pass standard STOPWORDS for common English words
    wc = WordCloud(
        width=1600, 
        height=800, 
        background_color='white', 
        stopwords=stopwords, 
        min_font_size=10,
        max_words=300,
        colormap='viridis'
    ).generate(text)
    
    # Plot
    plt.figure(figsize=(20, 10))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f'{conference.upper()} {year} Word Cloud', fontsize=20, pad=20)
    
    output_file = f'{conference}_{year}_wordcloud.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"Word cloud saved to {output_file}")
    plt.close()

def plot_categories(papers, conference, year):
    """
    Generate and save a pie chart of paper categories.
    """
    categories = [p['category'] for p in papers]
    category_counts = Counter(categories)
    category_totals = sum(category_counts.values())
    
    # Sort by count for better visualization
    sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [k for k, v in sorted_categories]
    sizes = [v for k, v in sorted_categories]
    
    # Configure font for Chinese support
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Increase figure size to accommodate legend
    plt.figure(figsize=(15, 12))
    
    # Create legend labels with percentages
    legend_labels = [f'{l} ({s/category_totals*100:.1f}%)' for l, s in zip(labels, sizes)]
    
    # Pie chart parameters
    # Remove labels from pie chart to avoid overlapping
    # Only show percentage on slices if > 2.0% to avoid clutter
    wedges, texts, autotexts = plt.pie(
        sizes, 
        labels=None, 
        autopct=lambda p: f'{p:.1f}%' if p > 2.0 else '', 
        startangle=140,
        pctdistance=0.85
    )
    
    # Improve label readability (Fonts bigger)
    plt.setp(autotexts, size=11, weight="bold")
    
    # Add legend to the right
    plt.legend(
        wedges, 
        legend_labels, 
        title="Categories", 
        loc="center left", 
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=12
    )
        
    plt.title(f'{conference.upper()} {year} Paper Categories Distribution ({category_totals} Papers In Total)', fontsize=18, pad=20)
    plt.axis('equal')
    
    output_file = f'{conference}_{year}_categories.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"Category distribution pie chart saved to {output_file}")
    plt.close()

def plot_top_authors(papers, conference, year, top_n=15):
    """
    Analyze and plot the most prolific authors.
    """
    print("Generating top authors chart...")
    # Configure font for Chinese support (if needed)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    all_authors = []
    for p in papers:
        if p.get('authors'):
            # Split by comma
            # Handle cases where authors might be empty or malformed
            authors = [a.strip() for a in p['authors'].split(',') if a.strip()]
            all_authors.extend(authors)
            
    if not all_authors:
        print("No author data available for plotting.")
        return

    # Count frequencies
    author_counts = Counter(all_authors)
    most_common = author_counts.most_common(top_n)
    
    if not most_common:
        print("No authors found.")
        return
        
    names = [name for name, count in most_common]
    counts = [count for name, count in most_common]
    
    # Plot
    plt.figure(figsize=(14, 10))
    # Horizontal bar chart
    y_pos = np.arange(len(names))
    bars = plt.barh(y_pos, counts, align='center', color='#4c72b0')
    plt.yticks(y_pos, names, fontsize=12)
    plt.gca().invert_yaxis()  # labels read top-to-bottom
    plt.xlabel('Number of Papers', fontsize=14)
    plt.title(f'Top {top_n} Authors - {conference.upper()} {year}', fontsize=18, pad=20)
    
    # Add counts at the end of bars
    for i, v in enumerate(counts):
        plt.text(v + 0.1, i, str(v), va='center', fontsize=11, fontweight='bold')
        
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    output_file = f'{conference}_{year}_top_authors.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"Top authors chart saved to {output_file}")
    plt.close()

def get_paper_category(title, abstract=""):
    """
    Classify the paper into a category based on its title and abstract.
    """
    title_lower = title.lower()
    abstract_lower = abstract.lower() if abstract else ""
    
    # Categories and their keywords
    # Priority order: specific domains -> tasks -> general methods
    categories = [
        ('数字人', [
            'avatar', 'human', 'face', 'facial', 'body', 'head', 'talking head', 'lip', 'gaze',
            'motion capture', 'animation', 'person re-identification', 're-id', 'expression',
            'pose estimation', 'hand', 'hair', 'garment', 'clothed', 'virtual try-on', 'skin'
        ]),
        ('3D视觉与重建', [
            '3d', 'reconstruction', 'nerf', 'gaussian splatting', 'gaussian', 'gs', 'point cloud',
            'mesh', 'stereo', 'depth', 'slam', 'structure from motion', 'sfm', 'view synthesis',
            'volumetric', 'lidar', 'occupancy', 'implicit field', 'radiance field', 'mvs', 
            'rendering', 'sdf', 'texture', 'surface', 'geometry', 'photometric', 'tri-plane'
        ]),
        ('自动驾驶', [
            'autonomous driving', 'self-driving', 'driving', 'vehicle', 'traffic', 'lane', 
            'pedestrian', 'waypoint', 'bev', 'bird\'s eye view', 'adas', 'autopilot', 
            'nuscenes', 'kitti', 'waymo', 'trajectory prediction', 'parking', 'road', 'urban'
        ]),
        ('医学影像与科学AI', [
            'medical', 'clinical', 'lesion', 'tumor', 'cancer', 'mri', 'ct', 'x-ray', 'covid',
            'pathology', 'histology', 'brain', 'diagnosis', 'biology', 'protein', 'disease',
            'molecule', 'drug', 'genome', 'climate', 'weather', 'physics', 'material', 'scientific',
            'polyp', 'organ', 'retinal', 'microscopy', 'cell', 'astronomy', 'chemistry'
        ]),
        ('具身智能与机器人', [
            'embodied', 'robot', 'manipulation', 'navigation', 'sim-to-real', 'teleoperation',
            'actuator', 'grasping', 'trajectory', 'locomotion', 'uav', 'drone', 'planning',
            'manipulator', 'quadruped', 'bipedal', 'dexterous', 'imitation learning', 'tactile'
        ]),
        ('推荐系统', [
            'recommendation', 'recommender', 'ranking', 'collaborative filtering', 'ctr', 
            'click-through', 'user behavior', 'preference', 'cold start', 'matrix factorization',
            'sequential recommendation', 'session-based', 'interest'
        ]),
        ('强化学习', [
            'reinforcement learning', 'rl', 'policy gradient', 'q-learning', 'actor-critic', 
            'exploration', 'reward', 'offline rl', 'inverse rl', 'decision making', 'control',
            'mdp', 'bandit', 'regret'
        ]),
        ('博弈论与多智能体', [
            'game theory', 'game', 'mechanism design', 'auction', 'fair division', 'equilibrium', 
            'nash', 'strategy', 'voting', 'social choice', 'multi-agent', 'marl', 'coordination', 
            'incentive', 'truthful', 'cooperative', 'competitive', 'zero-sum', 'persuasion'
        ]),
        ('逻辑、推理与知识表示', [
            'knowledge representation', 'reasoning', 'logic', 'logical', 'satisfiability', 'sat', 
            'csp', 'answer set', 'ontology', 'knowledge graph', 'deduction', 'formula', 'proof', 
            'solver', 'conjunctive query', 'rule', 'belief', 'symbolic', 'causal', 'causality'
        ]),
        ('神经科学与类脑AI', [
            'neuroscience', 'brain', 'neuron', 'spiking', 'snn', 'neuromorphic', 'cortex', 
            'neural code', 'fmri', 'eeg', 'synapse', 'biological', 'cognitive', 'plasticity',
            'hippocampus', 'neural dynamics'
        ]),
        ('进化计算与搜索', [
            'evolutionary', 'genetic algorithm', 'search', 'heuristic', 'combinatorial', 
            'tsp', 'knapsack', 'path finding', 'swarm', 'pso', 'ant colony', 'black-box optimization'
        ]),
        ('图神经网络', [
            'graph', 'gnn', 'gcn', 'gat', 'link prediction', 'subgraph', 'heterogeneous graph',
            'hypergraph', 'network embedding', 'graph neural', 'graph convolutional', 'message passing'
        ]),
        ('图像与视频生成', [
            'generation', 'generator', 'diffusion', 'gan', 'synthesis', 'editing', 'inpainting',
            'style transfer', 'generative', 'animating', 'translation', 'image-to-image',
            'text-to-image', 'text-to-video', 'video generation', 'controlnet', 'lora', 
            'latent', 'flow matching', 'score-based', 'ddpm'
        ]),
        ('多模态与NLP', [
            'multimodal', 'vision-language', 'vlm', 'clip', 'llm', 'captioning', 
            'visual question answering', 'vqa', 'text-image', 'cross-modal', 'grounding', 
            'visual language', 'reasoning', 'prompt', 'language model', 'bert', 'gpt',
            'nlp', 'text', 'translation', 'summarization', 'dialogue', 'sentiment', 
            'retrieval', 'chain-of-thought', 'hallucination', 'large language model',
            'token', 'embedding', 'rag', 'context', 'instruction tuning'
        ]),
        ('音频与语音', [
            'audio', 'speech', 'sound', 'music', 'voice', 'asr', 'tts', 'acoustic', 'speaker',
            'hearing', 'binaural'
        ]),
        ('图像视频处理', [
            'denoising', 'restoration', 'super-resolution', 'super resolution', 'deblurring', 
            'enhancement', 'compression', 'rain', 'haze', 'low-light', 'isp', 'deraining', 
            'dehazing', 'interpolation', 'video coding', 'shadow removal', 'demosaicing',
            'colorization', 'quality assessment', 'iqa'
        ]),
        ('AI理论与基础算法', [
            'quantization', 'pruning', 'distillation', 'efficiency', 'acceleration', 
            'nas', 'architecture search', 'lightweight', 'mobile', 'latency', 'inference', 
            'compression', 'sparse', 'federated', 'optimization', 'convergence', 
            'generalization', 'gradient', 'stochastic', 'convex', 'theory', 'theoretical',
            'learning theory', 'complexity', 'stability', 'meta-learning', 'clustering',
            'few-shot', 'zero-shot', 'transfer learning', 'domain adaptation', 'regularization',
            'loss function', 'optimizer', 'backpropagation', 'transformer', 'attention', 'backbone',
            'kernel', 'gaussian process', 'bayesian', 'classification', 'regression', 'online learning'
        ]),
        ('计算摄影与传感', [
            'computational photography', 'camera', 'sensor', 'imaging', 'optics', 
            'light field', 'event camera', 'hyperspectral', 'lens', 'coded aperture',
            'time-of-flight', 'tof', 'hdr'
        ]),
        ('可信AI与评估', [
            'explainability', 'interpretability', 'evaluation', 'metric', 'benchmark', 
            'fairness', 'robustness', 'uncertainty', 'calibration', 'bias', 'attack', 
            'defense', 'adversarial', 'privacy', 'safety', 'alignment', 'trustworthy',
            'watermark', 'deepfake', 'fake', 'verification', 'anomaly detection', 'out-of-distribution',
            'backdoor', 'jailbreak', 'poisoning'
        ]),
        ('视觉感知与理解', [
            'segmentation', 'detection', 'recognition', 'object', 
            'instance', 'semantic', 'panoptic', 'tracking', 'understanding', 'scene graph', 
            'action recognition', 'counting', 'localization', 'saliency', 'video understanding',
            'yolo', 'r-cnn', 'detr', 'mask', 'bounding box', 'pose'
        ]),
        ('时间序列', [
            'time series', 'forecasting', 'temporal', 'series', 'multivariate'
        ])
    ]
    
    best_category = '其他'
    max_score = 0
    
    for category_name, keywords in categories:
        score = 0
        for keyword in keywords:
            # Use regex for word boundary matching to avoid partial matches
            # e.g. "ct" should not match "detection", "rl" should not match "world"
            # Escape keyword to handle special characters like '.' or '+'
            # Allow optional 's' or 'es' suffix for plurals
            pattern = r'(?:^|[^a-zA-Z0-9])' + re.escape(keyword) + r'(?:s|es)?(?:$|[^a-zA-Z0-9])'
            
            # Weighted scoring
            if re.search(pattern, title_lower):
                score += 5  # Title match has high weight
            
            if abstract_lower and re.search(pattern, abstract_lower):
                score += 1  # Abstract match has lower weight
                
        if score > max_score:
            max_score = score
            best_category = category_name
            
    return best_category

def get_conference_config(conference, year):
    """
    Returns the configuration for the specified conference and year.
    """
    base_urls = {
        'cvpr': "https://cvpr.thecvf.com",
        'iccv': "https://iccv.thecvf.com",
        'neurips': "https://neurips.cc",
        'eccv': "https://eccv.ecva.net",
        'aaai': "https://ojs.aaai.org",
    }
    
    if conference not in base_urls:
        raise ValueError(f"Unsupported conference: {conference}")
        
    if conference in ['cvpr', 'iccv']:
        # Use Open Access for CVPR and ICCV
        base_url = "https://openaccess.thecvf.com"
        # e.g. https://openaccess.thecvf.com/CVPR2025?day=all
        papers_url = f"{base_url}/{conference.upper()}{year}?day=all"
        # Filter for paper detail pages (HTML files)
        filter_pattern = "_paper.html"
        
        return {
            'base_url': base_url,
            'papers_url': papers_url,
            'filter_pattern': filter_pattern
        }
        
    if conference == 'eccv':
        # ECCV papers are on ecva.net
        base_url = "https://www.ecva.net"
        papers_url = "https://www.ecva.net/papers.php"
        # Filter for specific year, e.g. papers/eccv_2024/
        # Use specific 'html/' folder to avoid duplicates (PDFs, supplements)
        filter_pattern = f"papers/eccv_{year}/papers_ECCV/html/"
        
        return {
            'base_url': base_url,
            'papers_url': papers_url,
            'filter_pattern': filter_pattern
        }
        
    base_url = base_urls[conference]
    
    if conference == 'aaai':
        raise ValueError("AAAI scraping is no longer supported.")
    
    # Common pattern for other conferences (e.g. NeurIPS uses different logic usually, but let's keep fallback)
    # The original virtual/year/papers.html logic was for virtual sites.
    papers_url = f"{base_url}/virtual/{year}/papers.html"
    
    return {
        'base_url': base_url,
        'papers_url': papers_url,
        'filter_pattern': f"/virtual/{year}/poster/"
    }

def get_paper_details(url, headers):
    """
    Fetch paper details including abstract and authors from the detail page.
    """
    try:
        # Add a small delay to be polite
        # time.sleep(0.1) 
        
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code != 200:
            return {'abstract': '', 'authors': '', 'pdf_url': ''}
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Abstract
        abstract = ""
        # Strategy 1: Look for meta description (often contains abstract)
        meta_desc = soup.find('meta', attrs={'name': 'citation_abstract'})
        if meta_desc:
            abstract = meta_desc['content']
        
        # Strategy 2: Look for div with class 'abstract' or id 'abstract'
        if not abstract:
            abs_div = soup.find('div', id='abstract') or soup.find('div', class_='abstract') or soup.find('p', class_='abstract')
            if abs_div:
                abstract = abs_div.get_text(strip=True)
        
        # Strategy 3: Look for header "Abstract" and take content
        if not abstract:
             headers_tags = soup.find_all(['h2', 'h3', 'h4', 'h5'])
             for h in headers_tags:
                 if 'abstract' in h.get_text(strip=True).lower():
                     sibling = h.find_next_sibling()
                     if sibling:
                         abstract = sibling.get_text(strip=True)
                     break

        # Authors
        authors = []
        # Strategy 1: Look for meta authors (most reliable for citation-ready sites)
        meta_authors = soup.find_all('meta', attrs={'name': 'citation_author'})
        if meta_authors:
            for m in meta_authors:
                name = m['content']
                # Normalize "Surname, Givenname" to "Givenname Surname"
                if ',' in name:
                    parts = name.split(',', 1)
                    if len(parts) == 2:
                        name = f"{parts[1].strip()} {parts[0].strip()}"
                authors.append(name)
        
        # Strategy 2: Look for common author containers
        if not authors:
            # NeurIPS virtual site often uses 'event-organizers' class for authors
            organizers_div = soup.find('div', class_='event-organizers')
            if organizers_div:
                text = organizers_div.get_text(strip=True)
                # Authors are often separated by '·'
                if '·' in text:
                    authors = [a.strip() for a in text.split('·')]
                else:
                    authors = [text]
            
            # Fallback to standard 'authors' class/id
            if not authors:
                auth_div = soup.find('div', class_='authors') or soup.find('ul', class_='authors') or soup.find('div', id='authors')
                if auth_div:
                    # Check for <i> tags which are common in CVF Open Access
                    i_tags = auth_div.find_all('i')
                    if i_tags:
                         authors = [i.get_text(strip=True) for i in i_tags]
                         # Split by comma if multiple authors in one <i> (rare but possible)
                         if len(authors) == 1 and ',' in authors[0]:
                             authors = [x.strip() for x in authors[0].split(',')]
                    else:
                        authors = [a.get_text(strip=True) for a in auth_div.find_all(['a', 'li'])]
                        if not authors: # If no tags, just text
                             # Split by comma
                             raw_text = auth_div.get_text(strip=True)
                             # Clean up common artifacts like trailing semicolon or asterisks
                             raw_text = raw_text.replace('*', '').replace(';', '')
                             authors = [x.strip() for x in raw_text.split(',') if x.strip()]

        # PDF URL
        pdf_url = ""
        # Strategy 1: Meta tag
        meta_pdf = soup.find('meta', attrs={'name': 'citation_pdf_url'})
        if meta_pdf:
            pdf_url = meta_pdf['content']
            
        # Strategy 2: Link with PDF text or .pdf href
        if not pdf_url:
            pdf_link = soup.find('a', href=re.compile(r'\.pdf$'))
            if not pdf_link:
                 pdf_link = soup.find('a', string=re.compile(r'PDF', re.I))
            
            if pdf_link and pdf_link.get('href'):
                pdf_url = pdf_link['href']
                if not pdf_url.startswith('http'):
                     pdf_url = urljoin(url, pdf_url)
                 
        return {
            'abstract': abstract.strip(),
            'authors': ", ".join(authors) if isinstance(authors, list) else str(authors),
            'pdf_url': pdf_url
        }
            
    except Exception as e:
        print(f"Error fetching details for {url}: {e}")
        return {'abstract': '', 'authors': '', 'pdf_url': ''}

def save_papers_to_csv(papers, conference, year):
    """
    Save papers to a CSV file.
    """
    output_file = f'{conference}_{year}_papers.csv'
    print(f"Saving {len(papers)} papers to {output_file}...")
    
    # Sort papers by category then by title
    papers.sort(key=lambda x: (x.get('category', '其他'), x.get('title', '')))
    
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
        fieldnames = ['id', 'title', 'category', 'type', 'url', 'pdf_url', 'abstract', 'authors', 'conference', 'year']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(papers)
    print(f"Done! Saved to {output_file}")
    return output_file

def load_papers_from_csv(conference, year):
    """
    Load papers from a CSV file.
    """
    input_file = f'{conference}_{year}_papers.csv'
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found.")
        return []
        
    print(f"Loading papers from {input_file}...")
    papers = []
    with open(input_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            papers.append(row)
            
    print(f"Loaded {len(papers)} papers.")
    return papers

import concurrent.futures
import time

def process_single_paper(a, base_url, headers, conference, year):
    """
    Process a single paper link: extract info, fetch details, classify.
    """
    try:
        title = a.get_text(strip=True)
        href = a.get('href')
        
        if not href:
            return None
        
        # Clean up ECCV login redirect
        if 'nextp=' in href:
            try:
                if '?' in href:
                    query = href.split('?', 1)[1]
                    parts = query.split('&')
                    for part in parts:
                        if part.startswith('nextp='):
                            next_val = part.split('=', 1)[1]
                            if next_val:
                                href = unquote(next_val)
                            break
            except:
                pass
        
        full_url = urljoin(base_url, href)
        
        # Extract ID from URL
        try:
            paper_id = href.rstrip('/').split('/')[-1]
            if not paper_id:
                 paper_id = href.rstrip('/').split('/')[-1]
        except:
            paper_id = ''
        
        # Fetch details
        details = get_paper_details(full_url, headers)
        
        # Determine category using title and abstract
        category = get_paper_category(title, details.get('abstract', ''))
        
        return {
            'id': paper_id,
            'title': title,
            'url': full_url,
            'pdf_url': details.get('pdf_url', ''), 
            'abstract': details.get('abstract', ''),
            'authors': details.get('authors', ''),
            'type': 'Poster', 
            'category': category,
            'conference': conference.upper(),
            'year': year
        }
    except Exception as e:
        print(f"Error processing paper {title[:30]}: {e}")
        return None

def scrape_papers(conference, year, max_workers=10):
    print(f"[{conference.upper()} {year}] Starting scraper with {max_workers} threads...")
    
    # Add User-Agent header to avoid 403 Forbidden
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        config = get_conference_config(conference, year)
        base_url = config['base_url']
        papers_url = config['papers_url']
        filter_pattern = config['filter_pattern']
        
        papers = []
        
        print(f"Fetching papers from {papers_url}...")
        response = requests.get(papers_url, headers=headers, timeout=30)
        
        if response.status_code != 200:
            print(f"Failed to fetch page: {response.status_code}")
            return []

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all paper links
        links = soup.find_all('a')
        paper_links = [a for a in links if filter_pattern in str(a.get('href', ''))]
        
        total = len(paper_links)
        print(f"Found {total} papers. Starting deep scrape (this may take a while)...")
        
        count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_paper = {executor.submit(process_single_paper, a, base_url, headers, conference, year): a for a in paper_links}
            
            for future in concurrent.futures.as_completed(future_to_paper):
                count += 1
                result = future.result()
                if result:
                    papers.append(result)
                
                # Print progress every 10 papers or so to reduce clutter
                if count % 10 == 0 or count == total:
                    print(f"Progress: [{count}/{total}] papers processed.")

        return papers
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return []

def reclassify_and_plot(conference, year):
    """
    Load papers from CSV, reclassify them using the latest logic, update CSV, and plot.
    """
    papers = load_papers_from_csv(conference, year)
    if not papers:
        return

    print("Reclassifying papers...")
    for p in papers:
        # Re-apply classification
        p['category'] = get_paper_category(p['title'], p.get('abstract', ''))
        
    # Save updated CSV
    save_papers_to_csv(papers, conference, year)
    
    # Plot
    try:
        plot_categories(papers, conference, year)
    except Exception as e:
        print(f"Failed to plot categories: {e}")
        import traceback
        traceback.print_exc()

    # Generate Word Cloud
    try:
        generate_wordcloud(papers, conference, year)
    except Exception as e:
        print(f"Failed to generate word cloud: {e}")
        import traceback
        traceback.print_exc()

    # Plot Top Authors
    try:
        plot_top_authors(papers, conference, year)
    except Exception as e:
        print(f"Failed to plot top authors: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Scrape papers from CVPR, ICCV, or NeurIPS.")
    
    parser.add_argument(
        '--conference', '-c', 
        type=str, 
        required=True, 
        choices=['cvpr', 'iccv', 'neurips', 'eccv'], 
        help="The conference to scrape (cvpr, iccv, neurips, eccv)"
    )
    
    parser.add_argument(
        '--year', '-y', 
        type=str, 
        required=True, 
        help="The year of the conference (e.g., 2025)"
    )
    
    parser.add_argument(
        '--mode', '-m',
        type=str,
        default='full',
        choices=['full', 'scrape', 'analyze'],
        help="Mode of operation: 'full' (scrape+analyze), 'scrape' (only scrape), 'analyze' (only classify+plot from existing CSV)"
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=20,
        help="Number of concurrent workers for scraping (default: 20)"
    )
    
    args = parser.parse_args()
    conference = args.conference.lower()
    year = args.year
    
    if args.mode in ['full', 'scrape']:
        papers = scrape_papers(conference, year, max_workers=args.workers)
        if papers:
            save_papers_to_csv(papers, conference, year)
            if args.mode == 'full':
                try:
                    plot_categories(papers, conference, year)
                except Exception as e:
                    print(f"Failed to plot categories: {e}")
                
                try:
                    generate_wordcloud(papers, conference, year)
                except Exception as e:
                    print(f"Failed to generate word cloud: {e}")
                
                try:
                    plot_top_authors(papers, conference, year)
                except Exception as e:
                    print(f"Failed to plot top authors: {e}")
    
    elif args.mode == 'analyze':
        reclassify_and_plot(conference, year)

if __name__ == "__main__":
    main()
