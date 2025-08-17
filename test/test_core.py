"""
Comprehensive test suite for Behavioral Boredom Index
Tests core functionality, privacy features, and ML models
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import asyncio
import torch

# Import modules to test
from bbi import (
    BoredomIndexEngine, 
    BoredomMonitoringAgent,
    PrivacyPreservingHash,
    NLPSentimentAnalyzer,
    BehavioralPatternExtractor,
    DeepLearningBoredomPredictor,
    FederatedBoredomClient,
    EngagementMetrics,
    BoredomSignal
)

class TestPrivacyPreservingHash:
    """Test privacy and security features"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.privacy_hasher = PrivacyPreservingHash()
    
    def test_employee_id_hashing(self):
        """Test employee ID hashing functionality"""
        employee_id = "emp_001"
        hashed_id = self.privacy_hasher.hash_employee_id(employee_id)
        
        # Test hash properties
        assert len(hashed_id) == 16
        assert hashed_id != employee_id
        
        # Test consistency
        hashed_id_2 = self.privacy_hasher.hash_employee_id(employee_id)
        assert hashed_id == hashed_id_2
        
        # Test different inputs produce different hashes
        different_hash = self.privacy_hasher.hash_employee_id("emp_002")
        assert hashed_id != different_hash
    
    def test_data_encryption(self):
        """Test sensitive data encryption"""
        sensitive_data = "confidential employee information"
        encrypted = self.privacy_hasher.encrypt_sensitive_data(sensitive_data)
        
        assert encrypted != sensitive_data
        assert len(encrypted) > len(sensitive_data)
    
    def test_hash_salt_rounds(self):
        """Test different salt rounds produce different hashes"""
        hasher_1 = PrivacyPreservingHash(salt_rounds=12)
        hasher_2 = PrivacyPreservingHash(salt_rounds=16)
        
        employee_id = "emp_test"
        hash_1 = hasher_1.hash_employee_id(employee_id)
        hash_2 = hasher_2.hash_employee_id(employee_id)
        
        assert hash_1 != hash_2


class TestNLPSentimentAnalyzer:
    """Test NLP and sentiment analysis functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.nlp_analyzer = NLPSentimentAnalyzer()
    
    def test_sentiment_analysis_positive(self):
        """Test positive sentiment detection"""
        positive_text = "I love working on this exciting project! It's innovative and challenging."
        results = self.nlp_analyzer.analyze_text_engagement(positive_text)
        
        assert 'sentiment' in results
        assert 'engagement' in results
        assert 'innovation' in results
        
        # Positive sentiment should be higher
        assert results['sentiment'] > 0.5
        assert results['innovation'] > 0.3  # Contains innovation keywords
    
    def test_sentiment_analysis_negative(self):
        """Test negative sentiment detection"""
        negative_text = "This work is boring and repetitive. I'm frustrated with the lack of challenge."
        results = self.nlp_analyzer.analyze_text_engagement(negative_text)
        
        assert 'sentiment' in results
        assert 'engagement' in results
        # Note: Negative sentiment might still show engagement due to strong emotion
    
    def test_empty_text_handling(self):
        """Test handling of empty or short text"""
        empty_results = self.nlp_analyzer.analyze_text_engagement("")
        short_results = self.nlp_analyzer.analyze_text_engagement("Hi")
        
        # Should return default values
        assert empty_results['sentiment'] == 0.5
        assert empty_results['engagement'] == 0.3
        assert empty_results['innovation'] == 0.2
        
        assert short_results['sentiment'] == 0.5
    
    def test_innovation_keyword_detection(self):
        """Test innovation keyword detection"""
        innovation_text = "We need to innovate and create breakthrough solutions to transform our approach."
        results = self.nlp_analyzer.analyze_text_engagement(innovation_text)
        
        assert results['innovation'] > 0.5


class TestBehavioralPatternExtractor:
    """Test behavioral pattern extraction and analysis"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.pattern_extractor = BehavioralPatternExtractor()
        
        # Create sample dataframes
        self.sample_email_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'response_time_hours': np.random.exponential(2, 30),
            'hour': np.random.randint(8, 18, 30),
            'weekday': np.tile(range(7), 5)[:30]
        })
        
        self.sample_calendar_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'duration_minutes': np.random.randint(30, 120, 30),
            'meeting_type': np.random.choice(['standup', 'project', 'one-on-one'], 30)
        })
        
        self.sample_document_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'edit_timestamp': pd.date_range('2024-01-01', periods=30, freq='D'),
            'document_type': np.random.choice(['doc', 'presentation', 'spreadsheet'], 30),
            'collaborators': np.random.randint(1, 5, 30)
        })
    
    def test_feature_extraction_complete_data(self):
        """Test feature extraction with complete data"""
        features = self.pattern_extractor.extract_digital_footprint_features(
            self.sample_email_data,
            self.sample_calendar_data,
            self.sample_document_data
        )
        
        # Check that features are extracted
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # Check for expected feature types
        email_features = [f for f in features.keys() if 'email' in f or 'response' in f]
        calendar_features = [f for f in features.keys() if 'calendar' in f or 'meeting' in f]
        document_features = [f for f in features.keys() if 'edit' in f or 'document' in f or 'collaboration' in f]
        
        assert len(email_features) > 0
        assert len(calendar_features) > 0
        assert len(document_features) > 0
    
    def test_empty_data_handling(self):
        """Test handling of empty dataframes"""
        empty_df = pd.DataFrame()
        
        features = self.pattern_extractor.extract_digital_footprint_features(
            empty_df, empty_df, empty_df
        )
        
        assert isinstance(features, dict)
        # Should handle empty data gracefully
    
    def test_email_pattern_analysis(self):
        """Test specific email pattern analysis"""
        features = self.pattern_extractor._analyze_email_patterns(self.sample_email_data)
        
        assert 'weekend_activity_ratio' in features
        assert 0 <= features['weekend_activity_ratio'] <= 1
        
        if 'avg_response_latency' in features:
            assert features['avg_response_latency'] >= 0
    
    def test_calendar_pattern_analysis(self):
        """Test calendar pattern analysis"""
        features = self.pattern_extractor._analyze_calendar_patterns(self.sample_calendar_data)
        
        assert 'calendar_emptiness' in features
        assert 0 <= features['calendar_emptiness'] <= 1


class TestDeepLearningBoredomPredictor:
    """Test deep learning model functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.model = DeepLearningBoredomPredictor(input_dim=50, hidden_dim=128)
    
    def test_model_initialization(self):
        """Test model initialization"""
        assert isinstance(self.model, torch.nn.Module)
        
        # Check layer dimensions
        assert self.model.input_layer.in_features == 50
        assert self.model.input_layer.out_features == 128
        assert self.model.output_layer.out_features == 3  # Low, Medium, High boredom
    
    def test_forward_pass(self):
        """Test model forward pass"""
        batch_size = 32
        input_dim = 50
        
        # Create random input
        x = torch.randn(batch_size, input_dim)
        
        # Forward pass
        output = self.model(x)
        
        # Check output shape and properties
        assert output.shape == (batch_size, 3)
        assert torch.allclose(output.sum(dim=1), torch.ones(batch_size), atol=1e-6)  # Softmax output
        assert (output >= 0).all() and (output <= 1).all()  # Valid probabilities
    
    def test_model_parameters(self):
        """Test model parameters"""
        params = list(self.model.parameters())
        assert len(params) > 0
        
        # Check parameter shapes
        for param in params:
            assert param.requires_grad


class TestFederatedBoredomClient:
    """Test federated learning client functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.model = DeepLearningBoredomPredictor(input_dim=10)
        self.train_data = np.random.randn(100, 10)
        self.train_labels = np.random.randint(0, 3, 100)
        self.client = FederatedBoredomClient(self.model, self.train_data, self.train_labels)
    
    def test_client_initialization(self):
        """Test federated client initialization"""
        assert self.client.model is not None
        assert self.client.train_data.shape == (100, 10)
        assert self.client.train_labels.shape == (100,)
    
    def test_get_parameters(self):
        """Test parameter extraction"""
        params = self.client.get_parameters(config={})
        
        assert isinstance(params, list)
        assert len(params) > 0
        assert all(isinstance(p, np.ndarray) for p in params)
    
    def test_set_parameters(self):
        """Test parameter setting"""
        original_params = self.client.get_parameters(config={})
        
        # Modify parameters
        modified_params = [p + 0.1 for p in original_params]
        self.client.set_parameters(modified_params)
        
        new_params = self.client.get_parameters(config={})
        
        # Check that parameters were updated
        for orig, new in zip(original_params, new_params):
            assert not np.allclose(orig, new)
    
    def test_fit_method(self):
        """Test local training (fit method)"""
        initial_params = self.client.get_parameters(config={})
        
        # Perform local training
        updated_params, num_examples, metrics = self.client.fit(initial_params, config={})
        
        assert isinstance(updated_params, list)
        assert num_examples == 100  # Size of training data
        assert isinstance(metrics, dict)


class TestBoredomIndexEngine:
    """Test main BoredomIndexEngine functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.engine = BoredomIndexEngine()
        
        # Sample data
        self.sample_email_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10, freq='D'),
            'response_time_hours': [1, 2, 1.5, 3, 2.5, 1.8, 2.2, 1.3, 2.8, 1.7],
            'hour': [9, 10, 14, 16, 11, 13, 15, 9, 17, 12],
            'weekday': [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        })
        
        self.sample_calendar_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10, freq='D'),
            'duration_minutes': [60, 90, 45, 120, 30, 75, 90, 60, 105, 45],
            'meeting_type': ['standup', 'project', 'one-on-one', 'project', 'standup', 
                           'project', 'one-on-one', 'standup', 'project', 'standup']
        })
        
        self.sample_document_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10, freq='D'),
            'edit_timestamp': pd.date_range('2024-01-01', periods=10, freq='D'),
            'document_type': ['doc', 'presentation', 'spreadsheet', 'doc', 'presentation',
                            'doc', 'spreadsheet', 'presentation', 'doc', 'spreadsheet'],
            'collaborators': [2, 3, 1, 4, 2, 3, 1, 2, 4, 2]
        })
    
    @pytest.mark.asyncio
    async def test_analyze_employee_engagement(self):
        """Test complete employee engagement analysis"""
        signal = await self.engine.analyze_employee_engagement(
            employee_id="test_emp_001",
            email_data=self.sample_email_data,
            calendar_data=self.sample_calendar_data,
            document_data=self.sample_document_data,
            communication_text="I'm working on exciting new projects and collaborating well with the team."
        )
        
        # Check BoredomSignal properties
        assert isinstance(signal, BoredomSignal)
        assert signal.employee_id is not None
        assert 0 <= signal.boredom_score <= 1
        assert 0 <= signal.confidence_level <= 1
        assert signal.risk_level in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        assert isinstance(signal.contributing_factors, dict)
        assert isinstance(signal.predicted_actions, list)
        assert isinstance(signal.timestamp, datetime)
    
    @pytest.mark.asyncio
    async def test_analyze_with_minimal_data(self):
        """Test analysis with minimal data"""
        empty_df = pd.DataFrame()
        
        signal = await self.engine.analyze_employee_engagement(
            employee_id="test_emp_minimal",
            email_data=empty_df,
            calendar_data=empty_df,
            document_data=empty_df,
            communication_text=""
        )
        
        # Should handle minimal data gracefully
        assert isinstance(signal, BoredomSignal)
        assert signal.boredom_score is not None
    
    def test_boredom_score_calculation(self):
        """Test boredom score calculation logic"""
        # Test with different feature combinations
        features_low_boredom = {
            'collaboration_level': 0.8,
            'sentiment': 0.7,
            'engagement': 0.8,
            'innovation': 0.6,
            'calendar_emptiness': 0.2
        }
        
        features_high_boredom = {
            'collaboration_level': 0.2,
            'sentiment': 0.3,
            'engagement': 0.2,
            'innovation': 0.1,
            'calendar_emptiness': 0.8,
            'avg_response_latency': 0.9
        }
        
        low_score = self.engine._calculate_boredom_score(features_low_boredom)
        high_score = self.engine._calculate_boredom_score(features_high_boredom)
        
        assert 0 <= low_score <= 1
        assert 0 <= high_score <= 1
        assert low_score < high_score  # High boredom features should yield higher score
    
    def test_risk_level_determination(self):
        """Test risk level classification"""
        assert self.engine._determine_risk_level(0.1) == "LOW"
        assert self.engine._determine_risk_level(0.4) == "MEDIUM"
        assert self.engine._determine_risk_level(0.7) == "HIGH"
        assert self.engine._determine_risk_level(0.9) == "CRITICAL"
    
    def test_team_insights_generation(self):
        """Test team-level insights generation"""
        # Create sample team signals
        team_signals = [
            BoredomSignal(
                employee_id="emp_001",
                boredom_score=0.3,
                confidence_level=0.8,
                contributing_factors={},
                risk_level="LOW",
                predicted_actions=[],
                timestamp=datetime.now()
            ),
            BoredomSignal(
                employee_id="emp_002",
                boredom_score=0.7,
                confidence_level=0.9,
                contributing_factors={'innovation_deficit': 0.8},
                risk_level="HIGH",
                predicted_actions=[],
                timestamp=datetime.now()
            ),
            BoredomSignal(
                employee_id="emp_003",
                boredom_score=0.5,
                confidence_level=0.85,
                contributing_factors={},
                risk_level="MEDIUM",
                predicted_actions=[],
                timestamp=datetime.now()
            )
        ]
        
        insights = self.engine.generate_team_insights(team_signals)
        
        assert insights['team_size'] == 3
        assert 0 <= insights['average_boredom_score'] <= 1
        assert 'risk_distribution' in insights
        assert 0 <= insights['turnover_risk_probability'] <= 1
        assert 0 <= insights['innovation_drought_risk'] <= 1
        assert isinstance(insights['recommendations'], list)
    
    def test_empty_team_insights(self):
        """Test team insights with empty data"""
        insights = self.engine.generate_team_insights([])
        assert 'error' in insights


class TestBoredomMonitoringAgent:
    """Test autonomous monitoring agent functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.engine = Mock(spec=BoredomIndexEngine)
        self.agent = BoredomMonitoringAgent(self.engine)
    
    def test_agent_initialization(self):
        """Test monitoring agent initialization"""
        assert self.agent.engine is self.engine
        assert not self.agent.monitoring_active
        assert isinstance(self.agent.alert_thresholds, dict)
    
    def test_stop_monitoring(self):
        """Test stopping the monitoring agent"""
        self.agent.monitoring_active = True
        self.agent.stop_monitoring()
        assert not self.agent.monitoring_active
    
    @pytest.mark.asyncio
    async def test_monitoring_cycle(self):
        """Test monitoring cycle execution"""
        # Mock the monitoring cycle method
        self.agent._perform_monitoring_cycle = AsyncMock()
        
        # Start monitoring for a very short time
        monitoring_task = asyncio.create_task(self.agent.start_monitoring(check_interval_hours=0.001))
        
        # Let it run briefly
        await asyncio.sleep(0.01)
        
        # Stop monitoring
        self.agent.stop_monitoring()
        
        # Wait for task to complete
        try:
            await asyncio.wait_for(monitoring_task, timeout=1.0)
        except asyncio.TimeoutError:
            monitoring_task.cancel()


class TestDataModels:
    """Test data model classes"""
    
    def test_engagement_metrics_creation(self):
        """Test EngagementMetrics dataclass"""
        metrics = EngagementMetrics(
            employee_id="test_001",
            timestamp=datetime.now(),
            email_response_latency=2.5,
            calendar_density=0.7,
            document_edit_frequency=5.0,
            collaboration_score=0.8,
            innovation_indicators=0.6,
            sentiment_score=0.75
        )
        
        assert metrics.employee_id == "test_001"
        assert metrics.email_response_latency == 2.5
        assert isinstance(metrics.focus_patterns, dict)
        assert isinstance(metrics.behavioral_anomalies, list)
    
    def test_boredom_signal_creation(self):
        """Test BoredomSignal dataclass"""
        signal = BoredomSignal(
            employee_id="test_001",
            boredom_score=0.45,
            confidence_level=0.87,
            contributing_factors={'factor1': 0.3, 'factor2': 0.6},
            risk_level="MEDIUM",
            predicted_actions=["action1", "action2"],
            timestamp=datetime.now()
        )
        
        assert signal.boredom_score == 0.45
        assert signal.risk_level == "MEDIUM"
        assert len(signal.predicted_actions) == 2


class TestIntegration:
    """Integration tests for complete workflows"""
    
    def setup_method(self):
        """Setup integration test fixtures"""
        self.engine = BoredomIndexEngine()
    
    @pytest.mark.asyncio
    async def test_end_to_end_analysis_workflow(self):
        """Test complete end-to-end analysis workflow"""
        # Prepare comprehensive test data
        email_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'response_time_hours': np.random.exponential(2, 30),
            'hour': np.random.randint(8, 18, 30),
            'weekday': np.tile(range(7), 5)[:30]
        })
        
        calendar_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'duration_minutes': np.random.randint(30, 180, 30),
            'meeting_type': np.random.choice(['standup', 'project', 'one-on-one', 'review'], 30)
        })
        
        document_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'edit_timestamp': pd.date_range('2024-01-01', periods=30, freq='D'),
            'document_type': np.random.choice(['doc', 'presentation', 'spreadsheet', 'code'], 30),
            'collaborators': np.random.randint(1, 6, 30)
        })
        
        communication_text = """
        I've been working on the new feature implementation and it's quite challenging.
        The team collaboration has been excellent, and I'm learning a lot from this project.
        Looking forward to the upcoming sprint and the innovative solutions we're developing.
        """
        
        # Perform analysis
        signal = await self.engine.analyze_employee_engagement(
            employee_id="integration_test_001",
            email_data=email_data,
            calendar_data=calendar_data,
            document_data=document_data,
            communication_text=communication_text
        )
        
        # Validate complete signal
        assert isinstance(signal, BoredomSignal)
        assert signal.employee_id is not None
        assert 0 <= signal.boredom_score <= 1
        assert signal.risk_level in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        assert len(signal.contributing_factors) > 0
        assert len(signal.predicted_actions) > 0
        assert signal.confidence_level > 0
    
    @pytest.mark.asyncio
    async def test_team_analysis_workflow(self):
        """Test team analysis workflow"""
        team_signals = []
        
        # Simulate analysis for multiple employees
        for i in range(5):
            signal = await self.engine.analyze_employee_engagement(
                employee_id=f"team_member_{i:03d}",
                email_data=pd.DataFrame(),
                calendar_data=pd.DataFrame(),
                document_data=pd.DataFrame(),
                communication_text=f"Sample communication for employee {i}"
            )
            team_signals.append(signal)
        
        # Generate team insights
        insights = self.engine.generate_team_insights(team_signals)
        
        assert insights['team_size'] == 5
        assert 'average_boredom_score' in insights
        assert 'risk_distribution' in insights
        assert 'recommendations' in insights
        assert len(insights['recommendations']) > 0


# Performance and load testing
class TestPerformance:
    """Performance and scalability tests"""
    
    def setup_method(self):
        """Setup performance test fixtures"""
        self.engine = BoredomIndexEngine()
    
    @pytest.mark.asyncio
    async def test_analysis_performance(self):
        """Test analysis performance with realistic data sizes"""
        import time
        
        # Create larger dataset
        large_email_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=365, freq='D'),
            'response_time_hours': np.random.exponential(2, 365),
            'hour': np.random.randint(8, 18, 365),
            'weekday': np.tile(range(7), 53)[:365]
        })
        
        start_time = time.time()
        
        signal = await self.engine.analyze_employee_engagement(
            employee_id="performance_test",
            email_data=large_email_data,
            calendar_data=pd.DataFrame(),
            document_data=pd.DataFrame(),
            communication_text="Performance test communication"
        )
        
        end_time = time.time()
        analysis_time = end_time - start_time
        
        # Should complete within reasonable time (< 5 seconds for 1 year of data)
        assert analysis_time < 5.0
        assert isinstance(signal, BoredomSignal)
    
    def test_memory_usage(self):
        """Test memory usage with large datasets"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large dataset
        large_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10000, freq='H'),
            'value': np.random.randn(10000)
        })
        
        # Process data
        features = self.engine.pattern_extractor.extract_digital_footprint_features(
            large_data, large_data, large_data
        )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 500MB for test)
        assert memory_increase < 500
        assert isinstance(features, dict)


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        "-v",
        "--cov=bbi",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--cov-fail-under=80",
        "--tb=short"
    ])
