import pandas as pd
import numpy as np
import nltk
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """
    Module for loading and preprocessing StackOverflow data
    """
    
    def __init__(self, train_path: str = "data/train.csv", test_path: str = "data/test.csv", analysis_dir: str = "analysis"):
        self.train_path = train_path
        self.test_path = test_path
        self.analysis_dir = Path(analysis_dir)
        self.train_data = None
        self.test_data = None
        self.combined_data = None
        
        # Create analysis directory if it doesn't exist
        self.analysis_dir.mkdir(exist_ok=True)
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def load_data(self, sample_size: Optional[int] = 1000) -> Dict[str, pd.DataFrame]:
        """
        Load and combine train and test data with optional sampling
        
        Args:
            sample_size: Number of samples to load from combined dataset (None for all)
            
        Returns:
            Dict containing train, test and combined DataFrames
        """
        # Load training data
        try:
            print(f"📥 Chargement de {self.train_path}...")
            self.train_data = pd.read_csv(self.train_path)
            print(f"✅ Train data: {self.train_data.shape[0]} lignes")
        except Exception as e:
            print(f"❌ Error loading train data: {e}")
            self.train_data = pd.DataFrame()
        
        # Charger les données de test
        try:
            print(f"📥 Chargement de {self.test_path}...")
            self.test_data = pd.read_csv(self.test_path)
            print(f"✅ Test data: {self.test_data.shape[0]} lignes")
        except Exception as e:
            print(f"❌ Error loading test data: {e}")
            self.test_data = pd.DataFrame()
        
        # Combiner les datasets
        try:
            print("🔄 Combinaison des datasets...")
            datasets_to_combine = []
            if not self.train_data.empty:
                train_subset = self.train_data.copy()
                train_subset['source'] = 'train'
                datasets_to_combine.append(train_subset)
            
            if not self.test_data.empty:
                test_subset = self.test_data.copy()
                test_subset['source'] = 'test'
                datasets_to_combine.append(test_subset)
            
            if datasets_to_combine:
                self.combined_data = pd.concat(datasets_to_combine, ignore_index=True)
                print(f"✅ Combined dataset: {self.combined_data.shape[0]} rows")
                
                # Appliquer l'échantillonnage si demandé
                if sample_size and sample_size < len(self.combined_data):
                    print(f"🎯 Échantillonnage: {sample_size} lignes")
                    self.combined_data = self.combined_data.sample(n=sample_size, random_state=42)
                    
            else:
                self.combined_data = pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Erreur lors de la combinaison: {e}")
            self.combined_data = pd.DataFrame()
            
        return {
            "train": self.train_data,
            "test": self.test_data,
            "combined": self.combined_data
        }
    
    def get_basic_stats(self) -> Dict[str, any]:
        """
        Get basic statistics of the combined dataset
        """
        if self.combined_data is None or self.combined_data.empty:
            self.load_data()
            
        stats = {
            "combined_shape": self.combined_data.shape if self.combined_data is not None else (0, 0),
            "train_shape": self.train_data.shape if self.train_data is not None else (0, 0),
            "test_shape": self.test_data.shape if self.test_data is not None else (0, 0),
            "columns": list(self.combined_data.columns) if self.combined_data is not None else [],
            "missing_values": self.combined_data.isnull().sum().to_dict() if self.combined_data is not None else {},
            "data_types": self.combined_data.dtypes.to_dict() if self.combined_data is not None else {},
            "source_distribution": self.combined_data['source'].value_counts().to_dict() if self.combined_data is not None else {}
        }
        
        return stats
    
    def analyze_text_length(self) -> pd.DataFrame:
        """
        Analyze text length (titles and bodies) on the combined dataset
        """
        if self.combined_data is None or self.combined_data.empty:
            self.load_data()
        
        analysis_data = []
        
        for _, row in self.combined_data.iterrows():
            # Analyser le titre
            title_words = len(str(row['Title']).split())
            title_chars = len(str(row['Title']))
            
            # Analyser le corps
            body_words = len(str(row['Body']).split())
            body_chars = len(str(row['Body']))
            
            analysis_data.append({
                'Id': row['Id'],
                'title_words': title_words,
                'title_chars': title_chars,
                'body_words': body_words,
                'body_chars': body_chars,
                'total_words': title_words + body_words,
                'total_chars': title_chars + body_chars,
                'score': row.get('Score', 0),
                'view_count': row.get('ViewCount', 0),
                'answer_count': row.get('AnswerCount', 0)
            })
        
        return pd.DataFrame(analysis_data)
    
    def get_word_frequency(self, text_column: str = 'Title', top_n: int = 100) -> Dict[str, int]:
        """
        Analyze word frequency in a given column on the combined dataset
        
        Args:
            text_column: Column to analyze ('Title' or 'Body')
            top_n: Nombre de mots les plus fréquents à retourner
            
        Returns:
            Dictionnaire des mots et leurs fréquences
        """
        if self.combined_data is None or self.combined_data.empty:
            self.load_data()
            
        from collections import Counter
        from nltk.corpus import stopwords
        import re
        
        stop_words = set(stopwords.words('english'))
        
        # Concaténer tous les textes
        all_text = ' '.join(self.combined_data[text_column].astype(str))
        
        # Nettoyer et tokeniser
        words = re.findall(r'\b[a-zA-Z]{2,}\b', all_text.lower())
        
        # Filtrer les stop words
        filtered_words = [word for word in words if word not in stop_words]
        
        # Compter les fréquences
        word_freq = Counter(filtered_words)
        
        return dict(word_freq.most_common(top_n))
    
    def get_tags_analysis(self) -> Dict[str, any]:
        """
        Analyze question tags on the combined dataset
        """
        if self.combined_data is None or self.combined_data.empty:
            self.load_data()
            
        from collections import Counter
        
        # Extraire tous les tags
        all_tags = []
        tag_counts = []
        
        for _, row in self.combined_data.iterrows():
            if pd.notna(row.get('TagsList', '')):
                try:
                    # Traiter TagsList qui peut être une string représentant une liste
                    tags_str = str(row['TagsList'])
                    if tags_str.startswith('[') and tags_str.endswith(']'):
                        # Retirer les crochets et diviser par les virgules
                        tags = [tag.strip().strip("'\"") for tag in tags_str[1:-1].split(',')]
                        tags = [tag for tag in tags if tag and tag != '']
                    else:
                        tags = [tags_str]
                    
                    all_tags.extend(tags)
                    tag_counts.append(len(tags))
                except:
                    tag_counts.append(0)
            else:
                tag_counts.append(0)
        
        # Statistiques des tags
        tag_freq = Counter(all_tags)
        
        return {
            'total_unique_tags': len(tag_freq),
            'total_tag_occurrences': len(all_tags),
            'avg_tags_per_question': np.mean(tag_counts),
            'tag_distribution': tag_counts,
            'most_common_tags': dict(tag_freq.most_common(50)),
            'tag_frequency': tag_freq
        }
    
    def get_correlation_data(self) -> pd.DataFrame:
        """
        Get data for correlation analysis on the combined dataset
        """
        text_analysis = self.analyze_text_length()
        
        # Ajouter les données de tags
        tags_data = []
        for _, row in self.combined_data.iterrows():
            try:
                tags_str = str(row.get('TagsList', '[]'))
                if tags_str.startswith('[') and tags_str.endswith(']'):
                    tags = [tag.strip().strip("'\"") for tag in tags_str[1:-1].split(',')]
                    tag_count = len([tag for tag in tags if tag and tag != ''])
                else:
                    tag_count = 1 if tags_str != '[]' and tags_str != '' else 0
                
                tags_data.append(tag_count)
            except:
                tags_data.append(0)
        
        text_analysis['tag_count'] = tags_data
        
        return text_analysis
    
    def save_analysis_results(self, analysis_data: Dict[str, any]) -> str:
        """
        Save analysis results in the analysis folder
        
        Args:
            analysis_data: Dictionary containing all analysis data
            
        Returns:
            Chemin vers le fichier de sauvegarde
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Préparer les données à sauvegarder
        save_data = {
            'metadata': {
                'timestamp': timestamp,
                'total_questions': analysis_data.get('basic_stats', {}).get('combined_shape', [0])[0],
                'sample_size': analysis_data.get('sample_size', 'unknown'),
                'source_distribution': analysis_data.get('basic_stats', {}).get('source_distribution', {})
            },
            'basic_stats': analysis_data.get('basic_stats', {}),
            'text_statistics': {
                'avg_title_words': float(analysis_data.get('text_analysis', pd.DataFrame()).get('title_words', pd.Series()).mean()) if not analysis_data.get('text_analysis', pd.DataFrame()).empty else 0,
                'avg_body_words': float(analysis_data.get('text_analysis', pd.DataFrame()).get('body_words', pd.Series()).mean()) if not analysis_data.get('text_analysis', pd.DataFrame()).empty else 0,
                'avg_total_words': float(analysis_data.get('text_analysis', pd.DataFrame()).get('total_words', pd.Series()).mean()) if not analysis_data.get('text_analysis', pd.DataFrame()).empty else 0
            },
            'word_frequency': {
                'top_title_words': analysis_data.get('word_freq_title', {}),
                'top_body_words': analysis_data.get('word_freq_body', {})
            },
            'tags_analysis': analysis_data.get('tags_analysis', {})
        }
        
        # Sauvegarder en JSON
        json_file = self.analysis_dir / f"analysis_results_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
        
        # Sauvegarder les DataFrames en pickle pour préserver les types
        pickle_file = self.analysis_dir / f"analysis_dataframes_{timestamp}.pkl"
        dataframe_data = {
            'text_analysis': analysis_data.get('text_analysis', pd.DataFrame()),
            'correlation_data': analysis_data.get('correlation_data', pd.DataFrame())
        }
        
        with open(pickle_file, 'wb') as f:
            pickle.dump(dataframe_data, f)
        
        # Créer un fichier de résumé lisible
        summary_file = self.analysis_dir / f"analysis_summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Analyse Exploratoire StackOverflow - {timestamp}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Dataset:\n")
            f.write(f"  - Total questions: {save_data['metadata']['total_questions']:,}\n")
            f.write(f"  - Échantillon: {save_data['metadata']['sample_size']}\n")
            f.write(f"  - Distribution sources: {save_data['metadata']['source_distribution']}\n\n")
            
            f.write(f"Statistiques textuelles:\n")
            f.write(f"  - Mots moyens titre: {save_data['text_statistics']['avg_title_words']:.1f}\n")
            f.write(f"  - Mots moyens corps: {save_data['text_statistics']['avg_body_words']:.1f}\n\n")
            
            f.write(f"Tags:\n")
            f.write(f"  - Tags uniques: {save_data['tags_analysis'].get('total_unique_tags', 0):,}\n")
            f.write(f"  - Tags par question (moyenne): {save_data['tags_analysis'].get('avg_tags_per_question', 0):.1f}\n\n")
            
            f.write("Top 10 mots titres:\n")
            for word, freq in list(save_data['word_frequency']['top_title_words'].items())[:10]:
                f.write(f"  - {word}: {freq}\n")
            
            f.write("\nTop 10 tags:\n")
            for tag, freq in list(save_data['tags_analysis'].get('most_common_tags', {}).items())[:10]:
                f.write(f"  - {tag}: {freq}\n")
        
        print(f"✅ Analysis saved:")
        print(f"  - JSON: {json_file}")
        print(f"  - DataFrames: {pickle_file}")
        print(f"  - Summary: {summary_file}")
        
        return str(json_file)
    
    def list_saved_analyses(self) -> List[Dict[str, str]]:
        """
        List saved analyses
        
        Returns:
            List of analyses with their metadata
        """
        analyses = []
        
        for json_file in self.analysis_dir.glob("analysis_results_*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                analyses.append({
                    'file': str(json_file),
                    'timestamp': data['metadata']['timestamp'],
                    'total_questions': data['metadata']['total_questions'],
                    'sample_size': data['metadata']['sample_size']
                })
                
            except Exception as e:
                print(f"Erreur lecture {json_file}: {e}")
                
        return sorted(analyses, key=lambda x: x['timestamp'], reverse=True)
    
    def load_latest_analysis(self) -> Optional[Dict[str, Union[str, int, pd.DataFrame, Dict]]]:
        """
        Load the latest saved analysis if one exists
        
        Returns:
            Dictionary with analysis data or None if no analysis found
        """
        analyses = self.list_saved_analyses()
        
        if not analyses:
            return None
            
        # Prendre la plus récente
        latest = analyses[0]
        
        try:
            # Charger le fichier JSON
            with open(latest['file'], 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Charger les DataFrames
            pickle_file = latest['file'].replace('analysis_results_', 'analysis_dataframes_').replace('.json', '.pkl')
            with open(pickle_file, 'rb') as f:
                dataframe_data = pickle.load(f)
            
            # Recombiner les données dans le format attendu
            analysis_data = {
                'basic_stats': json_data.get('basic_stats', {}),
                'text_analysis': dataframe_data.get('text_analysis', pd.DataFrame()),
                'word_freq_title': json_data.get('word_frequency', {}).get('top_title_words', {}),
                'word_freq_body': json_data.get('word_frequency', {}).get('top_body_words', {}),
                'tags_analysis': json_data.get('tags_analysis', {}),
                'sample_size': json_data.get('metadata', {}).get('sample_size', 'unknown'),
                'timestamp': json_data.get('metadata', {}).get('timestamp', 'unknown')
            }
            
            print(f"✅ Analysis loaded: {latest['timestamp']} ({latest['total_questions']} questions)")
            return analysis_data
            
        except Exception as e:
            print(f"❌ Erreur chargement analyse {latest['timestamp']}: {e}")
            return None
    
    def generate_full_analysis(self) -> Dict[str, any]:
        """
        Generate a complete analysis on the entire dataset
        
        Returns:
            Dictionary with all analysis data
        """
        print("🔄 Generating new analysis on complete dataset...")
        
        # Charger toutes les données (sample_size=None pour tout le dataset)
        data = self.load_data(sample_size=None)
        
        if data.get('combined') is None or data['combined'].empty:
            raise ValueError("Unable to load combined dataset")
        
        print(f"📊 Analyse en cours sur {data['combined'].shape[0]} questions...")
        
        # Générer toutes les analyses
        analysis_data = {
            'basic_stats': self.get_basic_stats(),
            'text_analysis': self.analyze_text_length(),
            'word_freq_title': self.get_word_frequency('Title', top_n=100),
            'word_freq_body': self.get_word_frequency('Body', top_n=100),
            'tags_analysis': self.get_tags_analysis(),
            'sample_size': data['combined'].shape[0]
        }
        
        # Sauvegarder automatiquement
        saved_file = self.save_analysis_results(analysis_data)
        print(f"✅ New analysis generated and saved: {saved_file}")
        
        return analysis_data
