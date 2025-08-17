from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import asyncio
import json
import logging
from contextlib import asynccontextmanager

from bbi import BoredomIndexEngine, BoredomMonitoringAgent, BoredomSignal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
bbi_engine = None
monitoring_agent = None
security = HTTPBearer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global bbi_engine, monitoring_agent
    
    # Startup
    logger.info("Initializing Behavioral Boredom Index API...")
    bbi_engine = BoredomIndexEngine()
    monitoring_agent = BoredomMonitoringAgent(bbi_engine)
    
    # Start background monitoring
    asyncio.create_task(monitoring_agent.start_monitoring(check_interval_hours=6))
    
    logger.info("BBI API initialized successfully")
    yield
    
    # Shutdown
    if monitoring_agent:
        monitoring_agent.stop_monitoring()
    logger.info("BBI API shutdown complete")

# FastAPI app initialization
app = FastAPI(
    title="Behavioral Boredom Index API",
    description="Privacy-preserving employee engagement analytics",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class EmailMetrics(BaseModel):
    """Email behavior metrics"""
    response_time_hours: List[float]
    email_count_daily: List[int]
    after_hours_ratio: float
    weekend_activity_ratio: float
    
    @validator('response_time_hours')
    def validate_response_times(cls, v):
        if any(x < 0 for x in v):
            raise ValueError('Response times must be non-negative')
        return v

class CalendarMetrics(BaseModel):
    """Calendar utilization metrics"""
    meeting_duration_minutes: List[int]
    meeting_count_daily: List[int]
    calendar_density: float
    meeting_diversity_score: float
    
    @validator('calendar_density')
    def validate_density(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Calendar density must be between 0 and 1')
        return v

class DocumentMetrics(BaseModel):
    """Document collaboration metrics"""
    edit_frequency_daily: List[int]
    collaboration_score: float
    document_diversity: float
    version_control_activity: List[int]

class CommunicationData(BaseModel):
    """Communication text for NLP analysis"""
    text_content: str
    communication_type: str = "general"  # email, slack, teams, etc.
    language: str = "en"

class EmployeeAnalysisRequest(BaseModel):
    """Complete employee analysis request"""
    employee_id: str
    time_range_days: int = 30
    email_metrics: Optional[EmailMetrics] = None
    calendar_metrics: Optional[CalendarMetrics] = None
    document_metrics: Optional[DocumentMetrics] = None
    communication_data: Optional[CommunicationData] = None
    
    @validator('time_range_days')
    def validate_time_range(cls, v):
        if not 1 <= v <= 365:
            raise ValueError('Time range must be between 1 and 365 days')
        return v

class BoredomAnalysisResponse(BaseModel):
    """Boredom analysis response"""
    employee_id: str
    boredom_score: float
    confidence_level: float
    risk_level: str
    contributing_factors: Dict[str, float]
    predicted_actions: List[str]
    recommendations: List[str]
    timestamp: datetime
    analysis_duration_ms: float

class TeamAnalysisRequest(BaseModel):
    """Team-level analysis request"""
    team_id: str
    employee_requests: List[EmployeeAnalysisRequest]
    include_benchmarks: bool = True

class TeamInsightsResponse(BaseModel):
    """Team insights response"""
    team_id: str
    team_size: int
    average_boredom_score: float
    risk_distribution: Dict[str, int]
    turnover_risk_probability: float
    innovation_drought_risk: float
    recommendations: List[str]
    individual_analyses: List[BoredomAnalysisResponse]
    timestamp: datetime

class MonitoringConfig(BaseModel):
    """Monitoring configuration"""
    check_interval_hours: int = 24
    alert_thresholds: Dict[str, float] = {
        "individual_high_risk": 0.75,
        "team_turnover_risk": 0.4,
        "innovation_drought": 0.5
    }
    notification_webhook: Optional[str] = None

# Dependency for authentication (simplified for demo)
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate API token"""
    # In production, implement proper JWT validation
    if credentials.credentials != "bbi-demo-token-2024":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"user_id": "demo_user", "permissions": ["read", "write"]}

# Utility functions
def convert_metrics_to_dataframe(metrics: Union[EmailMetrics, CalendarMetrics, DocumentMetrics], 
                                metric_type: str) -> pd.DataFrame:
    """Convert Pydantic metrics to DataFrame"""
    if metric_type == "email" and isinstance(metrics, EmailMetrics):
        return pd.DataFrame({
            'date': pd.date_range(end=datetime.now(), periods=len(metrics.response_time_hours), freq='D'),
            'response_time_hours': metrics.response_time_hours,
            'hour': [9] * len(metrics.response_time_hours),  # Default business hours
            'weekday': [i % 7 for i in range(len(metrics.response_time_hours))]
        })
    elif metric_type == "calendar" and isinstance(metrics, CalendarMetrics):
        return pd.DataFrame({
            'date': pd.date_range(end=datetime.now(), periods=len(metrics.meeting_duration_minutes), freq='D'),
            'duration_minutes': metrics.meeting_duration_minutes,
            'meeting_type': ['project'] * len(metrics.meeting_duration_minutes)
        })
    elif metric_type == "document" and isinstance(metrics, DocumentMetrics):
        return pd.DataFrame({
            'date': pd.date_range(end=datetime.now(), periods=len(metrics.edit_frequency_daily), freq='D'),
            'edit_timestamp': pd.date_range(end=datetime.now(), periods=len(metrics.edit_frequency_daily), freq='D'),
            'document_type': ['doc'] * len(metrics.edit_frequency_daily),
            'collaborators': [2] * len(metrics.edit_frequency_daily)
        })
    
    return pd.DataFrame()

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """API health check"""
    return {
        "message": "Behavioral Boredom Index API",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", response_model=Dict[str, Union[str, bool]])
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "engine_initialized": bbi_engine is not None,
        "monitoring_active": monitoring_agent.monitoring_active if monitoring_agent else False,
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": 0  # Would track actual uptime in production
    }

@app.post("/analyze/employee", response_model=BoredomAnalysisResponse)
async def analyze_employee(
    request: EmployeeAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user)
):
    """Analyze individual employee engagement"""
    start_time = datetime.now()
    
    try:
        # Convert metrics to DataFrames
        email_df = convert_metrics_to_dataframe(request.email_metrics, "email") if request.email_metrics else pd.DataFrame()
        calendar_df = convert_metrics_to_dataframe(request.calendar_metrics, "calendar") if request.calendar_metrics else pd.DataFrame()
        document_df = convert_metrics_to_dataframe(request.document_metrics, "document") if request.document_metrics else pd.DataFrame()
        
        # Extract communication text
        communication_text = request.communication_data.text_content if request.communication_data else ""
        
        # Perform analysis
        signal = await bbi_engine.analyze_employee_engagement(
            employee_id=request.employee_id,
            email_data=email_df,
            calendar_data=calendar_df,
            document_data=document_df,
            communication_text=communication_text
        )
        
        # Calculate analysis duration
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Generate recommendations based on risk level
        recommendations = generate_individual_recommendations(signal)
        
        # Log analysis for audit trail
        background_tasks.add_task(
            log_analysis_event,
            employee_id=request.employee_id,
            boredom_score=signal.boredom_score,
            risk_level=signal.risk_level,
            user_id=current_user["user_id"]
        )
        
        return BoredomAnalysisResponse(
            employee_id=signal.employee_id,
            boredom_score=signal.boredom_score,
            confidence_level=signal.confidence_level,
            risk_level=signal.risk_level,
            contributing_factors=signal.contributing_factors,
            predicted_actions=signal.predicted_actions,
            recommendations=recommendations,
            timestamp=signal.timestamp,
            analysis_duration_ms=duration_ms
        )
        
    except Exception as e:
        logger.error(f"Error analyzing employee {request.employee_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )

@app.post("/analyze/team", response_model=TeamInsightsResponse)
async def analyze_team(
    request: TeamAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user)
):
    """Analyze team-level engagement patterns"""
    start_time = datetime.now()
    
    try:
        # Analyze each team member
        individual_analyses = []
        team_signals = []
        
        for emp_request in request.employee_requests:
            # Convert metrics to DataFrames
            email_df = convert_metrics_to_dataframe(emp_request.email_metrics, "email") if emp_request.email_metrics else pd.DataFrame()
            calendar_df = convert_metrics_to_dataframe(emp_request.calendar_metrics, "calendar") if emp_request.calendar_metrics else pd.DataFrame()
            document_df = convert_metrics_to_dataframe(emp_request.document_metrics, "document") if emp_request.document_metrics else pd.DataFrame()
            
            communication_text = emp_request.communication_data.text_content if emp_request.communication_data else ""
            
            # Analyze individual
            signal = await bbi_engine.analyze_employee_engagement(
                employee_id=emp_request.employee_id,
                email_data=email_df,
                calendar_data=calendar_df,
                document_data=document_df,
                communication_text=communication_text
            )
            
            team_signals.append(signal)
            
            # Create individual response
            individual_analyses.append(BoredomAnalysisResponse(
                employee_id=signal.employee_id,
                boredom_score=signal.boredom_score,
                confidence_level=signal.confidence_level,
                risk_level=signal.risk_level,
                contributing_factors=signal.contributing_factors,
                predicted_actions=signal.predicted_actions,
                recommendations=generate_individual_recommendations(signal),
                timestamp=signal.timestamp,
                analysis_duration_ms=0  # Individual timing not tracked in team analysis
            ))
        
        # Generate team insights
        team_insights = bbi_engine.generate_team_insights(team_signals)
        
        # Log team analysis
        background_tasks.add_task(
            log_team_analysis_event,
            team_id=request.team_id,
            team_size=len(request.employee_requests),
            avg_boredom_score=team_insights["average_boredom_score"],
            user_id=current_user["user_id"]
        )
        
        return TeamInsightsResponse(
            team_id=request.team_id,
            team_size=team_insights["team_size"],
            average_boredom_score=team_insights["average_boredom_score"],
            risk_distribution=team_insights["risk_distribution"],
            turnover_risk_probability=team_insights["turnover_risk_probability"],
            innovation_drought_risk=team_insights["innovation_drought_risk"],
            recommendations=team_insights["recommendations"],
            individual_analyses=individual_analyses,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error analyzing team {request.team_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Team analysis failed: {str(e)}"
        )

@app.post("/monitoring/configure")
async def configure_monitoring(
    config: MonitoringConfig,
    current_user: Dict = Depends(get_current_user)
):
    """Configure monitoring parameters"""
    try:
        # Update monitoring agent configuration
        if monitoring_agent:
            monitoring_agent.alert_thresholds = config.alert_thresholds
            
        return {
            "message": "Monitoring configuration updated successfully",
            "config": config.dict(),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error configuring monitoring: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration failed: {str(e)}"
        )

@app.get("/monitoring/status")
async def get_monitoring_status(current_user: Dict = Depends(get_current_user)):
    """Get current monitoring status"""
    return {
        "monitoring_active": monitoring_agent.monitoring_active if monitoring_agent else False,
        "alert_thresholds": monitoring_agent.alert_thresholds if monitoring_agent else {},
        "last_check": datetime.now().isoformat(),
        "checks_performed": 0,  # Would track actual metrics in production
        "alerts_triggered": 0
    }

@app.post("/data/validate")
async def validate_data_quality(
    request: EmployeeAnalysisRequest,
    current_user: Dict = Depends(get_current_user)
):
    """Validate data quality and completeness"""
    validation_results = {
        "employee_id": request.employee_id,
        "data_quality_score": 0.0,
        "completeness_score": 0.0,
        "issues": [],
        "recommendations": []
    }
    
    completeness_factors = 0
    quality_factors = 0
    total_factors = 4  # email, calendar, document, communication
    
    # Validate email metrics
    if request.email_metrics:
        completeness_factors += 1
        if len(request.email_metrics.response_time_hours) >= 7:
            quality_factors += 1
        else:
            validation_results["issues"].append("Insufficient email data (need ≥7 days)")
    else:
        validation_results["issues"].append("Missing email metrics")
    
    # Validate calendar metrics
    if request.calendar_metrics:
        completeness_factors += 1
        if len(request.calendar_metrics.meeting_duration_minutes) >= 7:
            quality_factors += 1
        else:
            validation_results["issues"].append("Insufficient calendar data (need ≥7 days)")
    else:
        validation_results["issues"].append("Missing calendar metrics")
    
    # Validate document metrics
    if request.document_metrics:
        completeness_factors += 1
        if len(request.document_metrics.edit_frequency_daily) >= 7:
            quality_factors += 1
        else:
            validation_results["issues"].append("Insufficient document data (need ≥7 days)")
    else:
        validation_results["issues"].append("Missing document metrics")
    
    # Validate communication data
    if request.communication_data and len(request.communication_data.text_content) > 50:
        completeness_factors += 1
        quality_factors += 1
    elif request.communication_data:
        completeness_factors += 1
        validation_results["issues"].append("Communication text too short for reliable analysis")
    else:
        validation_results["issues"].append("Missing communication data")
    
    validation_results["completeness_score"] = completeness_factors / total_factors
    validation_results["data_quality_score"] = quality_factors / total_factors
    
    # Generate recommendations
    if validation_results["data_quality_score"] < 0.7:
        validation_results["recommendations"].append("Increase data collection period to at least 14 days")
    if validation_results["completeness_score"] < 0.8:
        validation_results["recommendations"].append("Enable additional data sources for comprehensive analysis")
    
    return validation_results

# Utility functions for background tasks
async def log_analysis_event(employee_id: str, boredom_score: float, risk_level: str, user_id: str):
    """Log analysis event for audit trail"""
    logger.info(f"Analysis performed - Employee: {employee_id}, Score: {boredom_score:.2f}, Risk: {risk_level}, User: {user_id}")

async def log_team_analysis_event(
