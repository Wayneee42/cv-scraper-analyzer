import requests
from bs4 import BeautifulSoup
import csv
import argparse
import sys
import os
import re
from urllib.parse import urljoin, unquote
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import concurrent.futures
import time
import traceback

class Config:
    @staticmethod
    def get(conference, year):
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
        papers_url = f"{base_url}/virtual/{year}/papers.html"
        
        return {
            'base_url': base_url,
            'papers_url': papers_url,
            'filter_pattern': f"/virtual/{year}/poster/"
        }

class PaperAnalyzer:
    CATEGORIES = [
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

    @classmethod
    def _compile_patterns(cls):
        """
        Lazy load and cache compiled regex patterns.
        """
        if not hasattr(cls, '_compiled_patterns'):
            cls._compiled_patterns = {}
            for category_name, keywords in cls.CATEGORIES:
                patterns = []
                for keyword in keywords:
                    # Use regex for word boundary matching
                    # Allow optional 's' or 'es' suffix for plurals
                    # Pre-compile with IGNORECASE
                    pattern = re.compile(r'(?:^|[^a-zA-Z0-9])' + re.escape(keyword) + r'(?:s|es)?(?:$|[^a-zA-Z0-9])', re.IGNORECASE)
                    patterns.append(pattern)
                cls._compiled_patterns[category_name] = patterns

    @classmethod
    def classify(cls, title, abstract=""):
        """
        Classify the paper into a category based on its title and abstract.
        """
        cls._compile_patterns()
        
        title_lower = title.lower()
        abstract_lower = abstract.lower() if abstract else ""
        
        best_category = '其他'
        max_score = 0
        
        for category_name, patterns in cls._compiled_patterns.items():
            score = 0
            for pattern in patterns:
                # Weighted scoring
                if pattern.search(title_lower):
                    score += 5  # Title match has high weight
                
                if abstract_lower and pattern.search(abstract_lower):
                    score += 1  # Abstract match has lower weight
                    
            if score > max_score:
                max_score = score
                best_category = category_name
                
        return best_category

class Visualizer:
    # Class-level constant for custom stopwords
    CUSTOM_STOPWORDS = {
        # General academic terms
        'paper', 'method', 'methods', 'methodology', 'propose', 'proposed', 'approach', 'model', 'framework',
        'pipeline', 'network', 'algorithm', 'architecture', 'module', 'component', 'system', 
        'scheme', 'strategy', 'mechanism', 'baseline', 'benchmark', 'dataset', 'across', 'without', 
        'data', 'database', 'corpus', 'collection', 'experiment', 'experiments', 'experimental', 
        'study', 'work', 'research', 'project', 'investigation', 'analysis', 'evaluation', 
        'ablation', 'comparison', 'discussion', 'introduction', 'conclusion', 'result', 'due', 
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

    @staticmethod
    def generate_wordcloud(papers, conference, year):
        """
        Generate and save a word cloud from paper abstracts.
        """
        print("Generating word cloud...")
        
        # Aggregate all abstracts
        text = " ".join([p.get('abstract', '') for p in papers if p.get('abstract')])
        
        if not text:
            print("No abstract text available for word cloud.")
            return None

        # Add custom stopwords
        stopwords = set(STOPWORDS)
        # Use regex to filter out custom stopwords and their plural forms (s/es)
        print("Filtering custom stopwords using regex (including plurals)...")
        sorted_stopwords = sorted(list(Visualizer.CUSTOM_STOPWORDS), key=len, reverse=True)
        
        for word in sorted_stopwords:
            # Pattern: boundary + word + optional s/es + boundary
            pattern = r'(?:^|[^a-zA-Z0-9])' + re.escape(word) + r'(?:s|es)?(?:$|[^a-zA-Z0-9])'
            text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
        
        # Generate word cloud
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
        
        folder = f"{conference}_{year}"
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        output_file = os.path.join(folder, f'{conference}_{year}_wordcloud.png')
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"Word cloud saved to {output_file}")
        plt.close()
        return output_file

    @staticmethod
    def plot_categories_interactive(papers, conference, year):
        """
        Generate a Plotly pie chart for paper categories.
        """
        try:
            import plotly.express as px
        except ImportError:
            print("Plotly not found. Please install it: pip install plotly")
            return None

        categories = [p['category'] for p in papers]
        category_counts = Counter(categories)
        
        df = pd.DataFrame(list(category_counts.items()), columns=['Category', 'Count'])
        
        fig = px.pie(
            df, 
            values='Count', 
            names='Category', 
            title=f'{conference.upper()} {year} Paper Categories',
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=800, width=1000)
        
        # Save as static image for folder storage (optional but requested)
        try:
            folder = f"{conference}_{year}"
            if not os.path.exists(folder):
                os.makedirs(folder)
            
            # Save as HTML for interactivity preservation
            html_file = os.path.join(folder, f'{conference}_{year}_categories.html')
            fig.write_html(html_file)
            
            # Also try to save as PNG using kaleido if available, otherwise skip
            # Note: kaleido might not be installed. 
            # Fallback: We can still call the matplotlib version to ensure PNG exists.
            # But the user specifically asked why images aren't updating. 
            # The issue is we are calling interactive version which returns a fig, 
            # but NOT calling the static matplotlib version anymore in streamlit_app.py
            # So we should call the static version here as a side effect or ensure both are generated.
            # Let's simply generate the static matplotlib version as well to ensure PNG exists.
            Visualizer.plot_categories(papers, conference, year)
            
        except Exception as e:
            print(f"Error saving static plots: {e}")
            
        return fig

    @staticmethod
    def plot_top_authors_interactive(papers, conference, year, top_n=15):
        """
        Generate a Plotly bar chart for top authors.
        """
        try:
            import plotly.express as px
        except ImportError:
            print("Plotly not found. Please install it: pip install plotly")
            return None
            
        all_authors = []
        for p in papers:
            if p.get('authors'):
                authors = [a.strip() for a in p['authors'].split(',') if a.strip()]
                all_authors.extend(authors)
                
        if not all_authors:
            return None
            
        author_counts = Counter(all_authors)
        most_common = author_counts.most_common(top_n)
        
        if not most_common:
            return None
            
        df = pd.DataFrame(most_common, columns=['Author', 'Papers'])
        df = df.sort_values(by='Papers', ascending=True) # Sort for barh (bottom to top)
        
        fig = px.bar(
            df, 
            x='Papers', 
            y='Author', 
            orientation='h',
            title=f'Top {top_n} Authors - {conference.upper()} {year}',
            text='Papers'
        )
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        fig.update_layout(height=600)
        
        # Save static files
        try:
            folder = f"{conference}_{year}"
            if not os.path.exists(folder):
                os.makedirs(folder)
                
            html_file = os.path.join(folder, f'{conference}_{year}_top_authors.html')
            fig.write_html(html_file)
            
            # Generate static PNG backup
            Visualizer.plot_top_authors(papers, conference, year, top_n)
        except Exception as e:
            print(f"Error saving static plots: {e}")
            
        return fig

    @staticmethod
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
        
        folder = f"{conference}_{year}"
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        output_file = os.path.join(folder, f'{conference}_{year}_categories.png')
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"Category distribution pie chart saved to {output_file}")
        plt.close()
        return output_file

    @staticmethod
    def plot_top_authors(papers, conference, year, top_n=15):
        """
        Analyze and plot the most prolific authors.
        """
        print("Generating top authors chart...")
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

        all_authors = []
        for p in papers:
            if p.get('authors'):
                # Split by comma, handling potential issues
                authors = [a.strip() for a in p['authors'].split(',') if a.strip()]
                all_authors.extend(authors)
                
        if not all_authors:
            print("No author data available for plotting.")
            return None

        # Count frequencies
        author_counts = Counter(all_authors)
        most_common = author_counts.most_common(top_n)
        
        if not most_common:
            print("No authors found.")
            return None
            
        names = [name for name, count in most_common]
        counts = [count for name, count in most_common]
        
        # Plot
        plt.figure(figsize=(14, 10))
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
        
        folder = f"{conference}_{year}"
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        output_file = os.path.join(folder, f'{conference}_{year}_top_authors.png')
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"Top authors chart saved to {output_file}")
        plt.close()
        return output_file

class DataManager:
    @staticmethod
    def save(papers, conference, year):
        """
        Save papers to a CSV file.
        """
        folder = f"{conference}_{year}"
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        output_file = os.path.join(folder, f'{conference}_{year}_papers.csv')
        print(f"Saving {len(papers)} papers to {output_file}...")
        
        # Sort papers by category then by title
        papers.sort(key=lambda x: (x.get('category', '其他'), x.get('title', '')))
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
                    fieldnames = ['id', 'title', 'category', 'type', 'url', 'pdf_url', 'abstract', 'authors', 'conference', 'year']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(papers)
                print(f"Done! Saved to {output_file}")
                return output_file
            except PermissionError:
                if attempt < max_retries - 1:
                    print(f"Permission denied accessing {output_file}. It might be open in another program.")
                    print("Waiting 5 seconds before retrying...")
                    time.sleep(5)
                else:
                    print(f"Error: Could not write to {output_file} after {max_retries} attempts.")
                    print("Please close the file if it is open and try again.")
                    # Try saving to a backup file
                    backup_file = os.path.join(folder, f'{conference}_{year}_papers_backup.csv')
                    print(f"Attempting to save to backup file: {backup_file}")
                    try:
                        with open(backup_file, 'w', newline='', encoding='utf-8-sig') as f:
                            fieldnames = ['id', 'title', 'category', 'type', 'url', 'pdf_url', 'abstract', 'authors', 'conference', 'year']
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(papers)
                        print(f"Successfully saved to {backup_file}")
                        return backup_file
                    except Exception as e:
                        print(f"Failed to save backup file: {e}")
                        raise

    @staticmethod
    def load(conference, year, file_path=None):
        """
        Load papers from a CSV file.
        """
        if file_path and os.path.exists(file_path):
            input_file = file_path
        else:
            filename = f'{conference}_{year}_papers.csv'
            folder = f"{conference}_{year}"
            
            # Try finding the file in the specific folder first
            folder_path = os.path.join(folder, filename)
            
            if os.path.exists(folder_path):
                input_file = folder_path
            elif os.path.exists(filename):
                # Fallback to root directory (legacy support)
                input_file = filename
            else:
                print(f"Error: File {filename} not found in {folder} or root.")
                return []
            
        print(f"Loading papers from {input_file}...")
        papers = []
        with open(input_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                papers.append(row)
                
        print(f"Loaded {len(papers)} papers.")
        return papers

class PaperScraper:
    def __init__(self, conference, year):
        self.conference = conference
        self.year = year
        self.config = Config.get(conference, year)
        self.base_url = self.config['base_url']
        self.papers_url = self.config['papers_url']
        self.filter_pattern = self.config['filter_pattern']
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        if self.conference == "neurips":
            self.max_detail_retries = 5
            self.request_delay = 0.2
            self.timeout = (10, 30)
        else:
            self.max_detail_retries = 3
            self.request_delay = 0
            self.timeout = 30

    def _get_details(self, url):
        """
        Fetch paper details including abstract and authors from the detail page.
        """
        try:
            response = None
            for attempt in range(self.max_detail_retries):
                try:
                    if self.request_delay:
                        time.sleep(self.request_delay)
                    response = self.session.get(url, timeout=self.timeout)
                    if response.status_code == 200:
                        break
                    if response.status_code in (429, 500, 502, 503, 504):
                        time.sleep(0.5 * (2 ** attempt))
                        continue
                    return {'abstract': '', 'authors': '', 'pdf_url': ''}
                except Exception:
                    if attempt == self.max_detail_retries - 1:
                        raise
                    time.sleep(0.5 * (2 ** attempt))
            if not response or response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Abstract
            abstract = ""
            # Strategy 1: Look for meta description
            meta_desc = soup.find('meta', attrs={'name': 'citation_abstract'})
            if meta_desc:
                abstract = meta_desc['content']
            
            # Strategy 2: Look for div with class 'abstract' or id 'abstract'
            if not abstract:
                abs_div = soup.find('div', id='abstract') or soup.find('div', class_='abstract') or soup.find('p', class_='abstract')
                if abs_div:
                    abstract = abs_div.get_text(strip=True)
            
            # Strategy 3: Look for header "Abstract"
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
            # Strategy 1: Look for meta authors
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
                             if len(authors) == 1 and ',' in authors[0]:
                                 authors = [x.strip() for x in authors[0].split(',')]
                        else:
                            authors = [a.get_text(strip=True) for a in auth_div.find_all(['a', 'li'])]
                            if not authors: 
                                 raw_text = auth_div.get_text(strip=True)
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
            return None

    def _process_paper(self, a):
        """
        Process a single paper link.
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
            
            full_url = urljoin(self.base_url, href)
            
            # Extract ID from URL
            try:
                paper_id = href.rstrip('/').split('/')[-1]
                if not paper_id:
                     paper_id = href.rstrip('/').split('/')[-1]
            except:
                paper_id = ''
            
            # Fetch details
            details = self._get_details(full_url)
            if details is None:
                return None
            
            # Determine category
            category = PaperAnalyzer.classify(title, details.get('abstract', ''))
            
            return {
                'id': paper_id,
                'title': title,
                'url': full_url,
                'pdf_url': details.get('pdf_url', ''), 
                'abstract': details.get('abstract', ''),
                'authors': details.get('authors', ''),
                'type': 'Poster', 
                'category': category,
                'conference': self.conference.upper(),
                'year': self.year
            }
        except Exception as e:
            print(f"Error processing paper {title[:30]}: {e}")
            return None

    def scrape(self, max_workers=10, progress_callback=None):
        print(f"[{self.conference.upper()} {self.year}] Starting scraper with {max_workers} threads...")
        
        try:
            papers = []
            
            print(f"Fetching papers from {self.papers_url}...")
            response = None
            for attempt in range(self.max_detail_retries):
                try:
                    if self.request_delay:
                        time.sleep(self.request_delay)
                    response = self.session.get(self.papers_url, timeout=self.timeout)
                    if response.status_code == 200:
                        break
                    if response.status_code in (429, 500, 502, 503, 504):
                        time.sleep(0.5 * (2 ** attempt))
                        continue
                    print(f"Failed to fetch page: {response.status_code}")
                    return []
                except Exception:
                    if attempt == self.max_detail_retries - 1:
                        raise
                    time.sleep(0.5 * (2 ** attempt))
            
            if not response or response.status_code != 200:
                print("Failed to fetch page after retries.")
                return []

            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all paper links
            links = soup.find_all('a')
            paper_links = [a for a in links if self.filter_pattern in str(a.get('href', ''))]
            
            total = len(paper_links)
            print(f"Found {total} papers. Starting deep scrape (this may take a while)...")
            if progress_callback:
                progress_callback(0, total, 0)
            
            count = 0
            error_count = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_paper = {executor.submit(self._process_paper, a): a for a in paper_links}
                
                for future in concurrent.futures.as_completed(future_to_paper):
                    count += 1
                    result = future.result()
                    if result:
                        papers.append(result)
                    else:
                        error_count += 1
                    if progress_callback:
                        progress_callback(count, total, error_count)
                    
                    if count % 10 == 0 or count == total:
                        print(f"Progress: [{count}/{total}] papers processed.")

            return papers
            
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            return []

class CVScraperApp:
    def __init__(self):
        self.parser = self._setup_argparser()

    def _setup_argparser(self):
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
        return parser

    def run(self):
        args = self.parser.parse_args()
        conference = args.conference.lower()
        year = args.year
        
        if args.mode in ['full', 'scrape']:
            scraper = PaperScraper(conference, year)
            papers = scraper.scrape(max_workers=args.workers)
            
            if papers:
                DataManager.save(papers, conference, year)
                if args.mode == 'full':
                    self._run_analysis(papers, conference, year)
        
        elif args.mode == 'analyze':
            self._reclassify_and_plot(conference, year)

    def _run_analysis(self, papers, conference, year):
        try:
            Visualizer.plot_categories(papers, conference, year)
        except Exception as e:
            print(f"Failed to plot categories: {e}")
        
        try:
            Visualizer.generate_wordcloud(papers, conference, year)
        except Exception as e:
            print(f"Failed to generate word cloud: {e}")
        
        try:
            Visualizer.plot_top_authors(papers, conference, year)
        except Exception as e:
            print(f"Failed to plot top authors: {e}")

    def _reclassify_and_plot(self, conference, year):
        """
        Load papers from CSV, reclassify them using the latest logic, update CSV, and plot.
        """
        papers = DataManager.load(conference, year)
        if not papers:
            return

        print("Reclassifying papers...")
        for p in papers:
            p['category'] = PaperAnalyzer.classify(p['title'], p.get('abstract', ''))
            
        DataManager.save(papers, conference, year)
        self._run_analysis(papers, conference, year)

if __name__ == "__main__":
    app = CVScraperApp()
    app.run()
