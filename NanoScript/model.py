import torch
import torch.nn as nn
import torch.nn.functional as F
from heartrate import HeartRateAnomalyDetector
from motion_classifier import MotionClassifier


class MultiModalFusionClassifier(nn.Module):
    
    def __init__(self, num_classes=4, fusion_mode='concatenate'):
        super(MultiModalFusionClassifier, self).__init__()
        
        # Pre-trained individual models (frozen)
        self.motion_encoder = MotionClassifier()
        self.physio_encoder = HeartRateAnomalyDetector(spo2_output=True)
        
        # Freeze pre-trained encoders
        for param in self.motion_encoder.parameters():
            param.requires_grad = False
        for param in self.physio_encoder.parameters():
            param.requires_grad = False
        
        # Extract richer embeddings from frozen models
        self.motion_feature_dim = 64
        self.physio_feature_dim = 64
        
        # Motion feature extractor (from 3-class output to embedding)
        self.motion_projector = nn.Sequential(
            nn.Linear(3, 32),  # 3 motion classes
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, self.motion_feature_dim),
            nn.ReLU()
        )
        
        # Physiological feature extractor (from anomaly + spo2 to embedding)
        self.physio_projector = nn.Sequential(
            nn.Linear(2, 32),  # HR anomaly + SpO2
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, self.physio_feature_dim),
            nn.ReLU()
        )
        
        # Learn how motion and physiology interact
        self.cross_attention = CrossModalAttention(
            motion_dim=self.motion_feature_dim,
            physio_dim=self.physio_feature_dim,
            hidden_dim=64
        )
        
        #Fusion Layer
        fusion_input_dim = self.motion_feature_dim + self.physio_feature_dim
        
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, num_classes)
        )
        
        # Specialized sub-network for critical conditions
        self.emergency_detector = EmergencyDetectionModule(
            motion_dim=self.motion_feature_dim,
            physio_dim=self.physio_feature_dim
        )
        
        self.num_classes = num_classes
        self.fusion_mode = fusion_mode
        
    def forward(self, motion_data, hr_data):
        """
        Forward pass through multi-modal fusion network
        
        Args:
            motion_data: (batch, 6, 100) - IMU sensor data
            hr_data: (batch, 100, 1) - Heart rate sequence
            
        Returns:
            logits: (batch, num_classes) - Class predictions
        """
        batch_size = motion_data.shape[0]
        
        # ==================== ENCODE MODALITIES ====================
        with torch.no_grad():
            # Motion encoding
            motion_logits = self.motion_encoder(motion_data)  # (batch, 3)
            motion_probs = F.softmax(motion_logits, dim=1)
            
            # Physiological encoding
            hr_anomaly, spo2 = self.physio_encoder(hr_data)  # (batch, 1), (batch, 1)
            physio_features = torch.cat([hr_anomaly, spo2], dim=1)  # (batch, 2)
        
        # ==================== PROJECT TO EMBEDDING SPACE ====================
        motion_embedding = self.motion_projector(motion_probs)  # (batch, 64)
        physio_embedding = self.physio_projector(physio_features)  # (batch, 64)
        
        # ==================== CROSS-MODAL ATTENTION ====================
        # Learn inter-modal relationships
        motion_attended, physio_attended = self.cross_attention(
            motion_embedding, physio_embedding
        )
        
        # ==================== FUSION ====================
        # Concatenate attended features
        fused_features = torch.cat([motion_attended, physio_attended], dim=1)
        
        # Main classification
        fusion_logits = self.fusion_network(fused_features)
        
        # ==================== EMERGENCY BOOSTING ====================
        # Detect critical conditions
        emergency_score = self.emergency_detector(
            motion_embedding, physio_embedding, spo2
        )
        
        # Boost emergency class when critical conditions detected
        emergency_boost = torch.zeros_like(fusion_logits)
        emergency_boost[:, 0] = emergency_score.squeeze() * 3.0
        
        final_logits = fusion_logits + emergency_boost
        
        return final_logits
    
    def get_interpretability_info(self, motion_data, hr_data):
        """
        Extract detailed information for model interpretability
        
        Returns:
            dict with all intermediate outputs and attention weights
        """
        with torch.no_grad():
            batch_size = motion_data.shape[0]
            
            # Encode modalities
            motion_logits = self.motion_encoder(motion_data)
            motion_probs = F.softmax(motion_logits, dim=1)
            hr_anomaly, spo2 = self.physio_encoder(hr_data)
            physio_features = torch.cat([hr_anomaly, spo2], dim=1)
            
            # Project to embeddings
            motion_embedding = self.motion_projector(motion_probs)
            physio_embedding = self.physio_projector(physio_features)
            
            # Get attention weights
            motion_attended, physio_attended, attn_weights = self.cross_attention.get_attention_weights(
                motion_embedding, physio_embedding
            )
            
            # Emergency score
            emergency_score = self.emergency_detector(
                motion_embedding, physio_embedding, spo2
            )
            
            # Final prediction
            fused = torch.cat([motion_attended, physio_attended], dim=1)
            logits = self.fusion_network(fused)
            probs = F.softmax(logits, dim=1)
            
            return {
                'motion_probs': motion_probs.cpu().numpy(),
                'hr_anomaly': hr_anomaly.cpu().numpy(),
                'spo2_normalized': spo2.cpu().numpy(),
                'spo2_percent': (spo2 * 30 + 70).cpu().numpy(),
                'motion_embedding': motion_embedding.cpu().numpy(),
                'physio_embedding': physio_embedding.cpu().numpy(),
                'motion_to_physio_attention': attn_weights['motion_to_physio'].cpu().numpy(),
                'physio_to_motion_attention': attn_weights['physio_to_motion'].cpu().numpy(),
                'emergency_score': emergency_score.cpu().numpy(),
                'emergency_gate': emergency_score.cpu().numpy(),
                'class_probs': probs.cpu().numpy()
            }
    
    def save_model(self, path):
        """Save fusion model weights"""
        torch.save(self.state_dict(), path)
        print(f"✓ Multi-modal fusion model saved to {path}")
    
    def load_model(self, path):
        """Load fusion model weights"""
        self.load_state_dict(torch.load(path, map_location='cpu'))
        self.eval()
        print(f"✓ Multi-modal fusion model loaded from {path}")


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism to learn inter-modal relationships
    
    Computes attention between motion and physiological features:
    - How does motion pattern influence physiology interpretation?
    - How does physiology influence motion pattern interpretation?
    """
    
    def __init__(self, motion_dim, physio_dim, hidden_dim=64):
        super(CrossModalAttention, self).__init__()
        
        # Motion attending to physiology
        self.motion_query = nn.Linear(motion_dim, hidden_dim)
        self.physio_key = nn.Linear(physio_dim, hidden_dim)
        self.physio_value = nn.Linear(physio_dim, physio_dim)
        
        # Physiology attending to motion
        self.physio_query = nn.Linear(physio_dim, hidden_dim)
        self.motion_key = nn.Linear(motion_dim, hidden_dim)
        self.motion_value = nn.Linear(motion_dim, motion_dim)
        
        self.scale = hidden_dim ** 0.5
        
    def forward(self, motion_features, physio_features):
        """
        Args:
            motion_features: (batch, motion_dim)
            physio_features: (batch, physio_dim)
            
        Returns:
            motion_attended: (batch, motion_dim)
            physio_attended: (batch, physio_dim)
        """
        # Motion attends to physiology
        m_query = self.motion_query(motion_features)  # (batch, hidden)
        p_key = self.physio_key(physio_features)      # (batch, hidden)
        p_value = self.physio_value(physio_features)  # (batch, physio_dim)
        
        attn_mp = F.softmax((m_query * p_key) / self.scale, dim=1)
        motion_context = attn_mp.unsqueeze(1) @ p_value.unsqueeze(2)  # Broadcast
        motion_attended = motion_features + motion_context.squeeze(1) * 0.5
        
        # Physiology attends to motion
        p_query = self.physio_query(physio_features)
        m_key = self.motion_key(motion_features)
        m_value = self.motion_value(motion_features)
        
        attn_pm = F.softmax((p_query * m_key) / self.scale, dim=1)
        physio_context = attn_pm.unsqueeze(1) @ m_value.unsqueeze(2)
        physio_attended = physio_features + physio_context.squeeze(1) * 0.5
        
        return motion_attended, physio_attended
    
    def get_attention_weights(self, motion_features, physio_features):
        """Get attention weights for interpretability"""
        m_query = self.motion_query(motion_features)
        p_key = self.physio_key(physio_features)
        
        attn_mp = F.softmax((m_query * p_key) / self.scale, dim=1)
        
        p_query = self.physio_query(physio_features)
        m_key = self.motion_key(motion_features)
        
        attn_pm = F.softmax((p_query * m_key) / self.scale, dim=1)
        
        motion_attended, physio_attended = self.forward(motion_features, physio_features)
        
        return motion_attended, physio_attended, {
            'motion_to_physio': attn_mp,
            'physio_to_motion': attn_pm
        }


class EmergencyDetectionModule(nn.Module):
    """
    Specialized module for detecting critical emergency conditions
    
    Focuses on specific patterns:
    - Low SpO2 (<85%) + Panic motion = Drowning
    - Low SpO2 + Immobile = Unconscious
    - Very high HR + Low SpO2 = Medical emergency
    """
    
    def __init__(self, motion_dim, physio_dim):
        super(EmergencyDetectionModule, self).__init__()
        
        self.emergency_network = nn.Sequential(
            nn.Linear(motion_dim + physio_dim + 1, 64),  # +1 for raw SpO2
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Critical SpO2 threshold (normalized)
        self.spo2_critical = 0.5  # ~85% = (85-70)/30
        
    def forward(self, motion_embedding, physio_embedding, spo2):
        """
        Args:
            motion_embedding: (batch, motion_dim)
            physio_embedding: (batch, physio_dim)
            spo2: (batch, 1) - normalized SpO2 [0, 1]
            
        Returns:
            emergency_score: (batch, 1) - probability of emergency
        """
        # Concatenate all features
        combined = torch.cat([motion_embedding, physio_embedding, spo2], dim=1)
        
        # Compute emergency score
        emergency_score = self.emergency_network(combined)
        
        # Boost score if SpO2 is critically low
        spo2_boost = (spo2 < self.spo2_critical).float() * 0.3
        emergency_score = torch.clamp(emergency_score + spo2_boost, 0.0, 1.0)
        
        return emergency_score


# Alias for backward compatibility
FusionClassifier = MultiModalFusionClassifier
