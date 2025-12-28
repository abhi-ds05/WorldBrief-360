"""
Reward and wallet management endpoints.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from app.core.database import get_db
from app.core.security import get_current_user
from app.db.models import User, Wallet, Transaction, Incident
from app.schemas import (
    WalletResponse,
    TransactionResponse,
    TransactionCreate,
    RewardHistoryResponse,
    LeaderboardEntry,
    PaginatedResponse,
)
from app.schemas.common import PaginationParams
from app.services.community.wallet_service import (
    get_user_wallet,
    create_transaction,
    get_transaction_history,
    calculate_user_rewards,
    check_eligible_for_reward,
)

router = APIRouter()


@router.get("/wallet", response_model=WalletResponse)
async def get_wallet(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> WalletResponse:
    """
    Get current user's wallet information.
    """
    wallet = await get_user_wallet(current_user.id, db)
    
    if not wallet:
        # Create wallet if it doesn't exist
        wallet = Wallet(user_id=current_user.id, balance=0)
        db.add(wallet)
        db.commit()
        db.refresh(wallet)
    
    return WalletResponse.from_orm(wallet)


@router.get("/transactions", response_model=PaginatedResponse[TransactionResponse])
async def get_transactions(
    pagination: PaginationParams = Depends(),
    transaction_type: Optional[str] = Query(None, description="Filter by transaction type"),
    start_date: Optional[datetime] = Query(None, description="Filter by start date"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> PaginatedResponse[TransactionResponse]:
    """
    Get transaction history for current user.
    """
    transactions = await get_transaction_history(
        user_id=current_user.id,
        transaction_type=transaction_type,
        start_date=start_date,
        end_date=end_date,
        limit=pagination.limit,
        offset=pagination.offset,
        db=db
    )
    
    total = db.query(Transaction).filter(
        Transaction.user_id == current_user.id
    ).count()
    
    if transaction_type:
        total = db.query(Transaction).filter(
            Transaction.user_id == current_user.id,
            Transaction.transaction_type == transaction_type
        ).count()
    
    return PaginatedResponse(
        items=[TransactionResponse.from_orm(tx) for tx in transactions],
        total=total,
        page=pagination.page,
        per_page=pagination.limit,
    )


@router.get("/transactions/{transaction_id}", response_model=TransactionResponse)
async def get_transaction(
    transaction_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TransactionResponse:
    """
    Get a specific transaction.
    """
    transaction = db.query(Transaction).filter(
        Transaction.id == transaction_id,
        Transaction.user_id == current_user.id,
    ).first()
    
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    return TransactionResponse.from_orm(transaction)


@router.post("/transactions", response_model=TransactionResponse)
async def create_new_transaction(
    transaction_data: TransactionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TransactionResponse:
    """
    Create a new transaction (e.g., purchase, transfer).
    """
    # Get user wallet
    wallet = await get_user_wallet(current_user.id, db)
    
    # Check balance for purchases
    if transaction_data.transaction_type == "purchase":
        if wallet.balance < transaction_data.amount:
            raise HTTPException(
                status_code=400,
                detail="Insufficient balance"
            )
    
    # Create transaction
    transaction = await create_transaction(
        user_id=current_user.id,
        transaction_data=transaction_data,
        db=db
    )
    
    # Update wallet balance
    if transaction_data.transaction_type == "purchase":
        wallet.balance -= transaction_data.amount
    elif transaction_data.transaction_type == "reward":
        wallet.balance += transaction_data.amount
    
    wallet.updated_at = datetime.utcnow()
    db.commit()
    
    return TransactionResponse.from_orm(transaction)


@router.get("/rewards/history", response_model=RewardHistoryResponse)
async def get_reward_history(
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RewardHistoryResponse:
    """
    Get reward history for current user.
    """
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Get reward transactions
    rewards = db.query(Transaction).filter(
        Transaction.user_id == current_user.id,
        Transaction.transaction_type == "reward",
        Transaction.created_at >= start_date,
    ).order_by(desc(Transaction.created_at)).all()
    
    # Calculate totals
    total_rewards = sum(tx.amount for tx in rewards)
    
    # Group by reward type
    rewards_by_type = {}
    for reward in rewards:
        reward_type = reward.metadata.get("reward_type", "other")
        if reward_type not in rewards_by_type:
            rewards_by_type[reward_type] = {
                "count": 0,
                "total_amount": 0
            }
        rewards_by_type[reward_type]["count"] += 1
        rewards_by_type[reward_type]["total_amount"] += reward.amount
    
    # Calculate daily earnings
    daily_earnings = []
    for i in range(days):
        date = datetime.utcnow() - timedelta(days=i)
        day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        
        day_rewards = db.query(func.sum(Transaction.amount)).filter(
            Transaction.user_id == current_user.id,
            Transaction.transaction_type == "reward",
            Transaction.created_at >= day_start,
            Transaction.created_at < day_end,
        ).scalar() or 0
        
        daily_earnings.append({
            "date": day_start.date().isoformat(),
            "amount": day_rewards
        })
    
    return RewardHistoryResponse(
        user_id=current_user.id,
        period_days=days,
        total_rewards=total_rewards,
        rewards_by_type=rewards_by_type,
        daily_earnings=daily_earnings,
        generated_at=datetime.utcnow(),
    )


@router.get("/leaderboard", response_model=List[LeaderboardEntry])
async def get_leaderboard(
    timeframe: str = Query("weekly", description="Timeframe: daily, weekly, monthly, alltime"),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
) -> List[LeaderboardEntry]:
    """
    Get user leaderboard based on rewards/contributions.
    """
    # Determine timeframe
    if timeframe == "daily":
        start_date = datetime.utcnow() - timedelta(days=1)
    elif timeframe == "weekly":
        start_date = datetime.utcnow() - timedelta(days=7)
    elif timeframe == "monthly":
        start_date = datetime.utcnow() - timedelta(days=30)
    else:  # alltime
        start_date = None
    
    # Query for leaderboard
    query = db.query(
        User.id,
        User.username,
        User.reputation_score,
        func.sum(Transaction.amount).label("total_rewards"),
        func.count(Transaction.id).label("transaction_count"),
    ).join(
        Transaction, Transaction.user_id == User.id
    ).filter(
        Transaction.transaction_type == "reward",
        User.is_active == True,
    )
    
    if start_date:
        query = query.filter(Transaction.created_at >= start_date)
    
    leaderboard_data = query.group_by(User.id)\
                           .order_by(desc("total_rewards"))\
                           .limit(limit)\
                           .all()
    
    # Add user rank
    leaderboard = []
    for rank, (user_id, username, reputation, total_rewards, tx_count) in enumerate(leaderboard_data, 1):
        leaderboard.append(LeaderboardEntry(
            rank=rank,
            user_id=user_id,
            username=username,
            reputation_score=reputation,
            total_rewards=total_rewards or 0,
            transaction_count=tx_count or 0,
            is_current_user=current_user and user_id == current_user.id
        ))
    
    return leaderboard


@router.get("/eligible/incident/{incident_id}")
async def check_incident_reward_eligibility(
    incident_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Check if user is eligible for reward for a specific incident.
    """
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    # Check if user is the reporter
    if incident.reporter_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="Only the incident reporter can check reward eligibility"
        )
    
    eligible, reason = await check_eligible_for_reward(
        user_id=current_user.id,
        incident_id=incident_id,
        db=db
    )
    
    return {
        "eligible": eligible,
        "reason": reason,
        "incident_id": incident_id,
        "incident_status": incident.verification_status,
        "coins_awarded": incident.coins_awarded,
    }


@router.post("/claim/incident/{incident_id}")
async def claim_incident_reward(
    incident_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Claim reward for a verified incident.
    """
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    # Check if user is the reporter
    if incident.reporter_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="Only the incident reporter can claim rewards"
        )
    
    # Check if already awarded
    if incident.coins_awarded:
        raise HTTPException(
            status_code=400,
            detail="Reward already claimed for this incident"
        )
    
    # Check if incident is verified
    if incident.verification_status != "verified":
        raise HTTPException(
            status_code=400,
            detail="Incident must be verified before claiming reward"
        )
    
    # Calculate reward amount based on incident severity and verification confidence
    base_reward = 50  # Base coins
    severity_multiplier = {
        "low": 0.5,
        "medium": 1.0,
        "high": 1.5,
        "critical": 2.0,
    }
    
    reward_amount = int(base_reward * severity_multiplier.get(incident.severity, 1.0))
    
    # Add bonus for high confidence
    if incident.confidence_score >= 0.8:
        reward_amount = int(reward_amount * 1.2)
    
    # Create reward transaction
    transaction = Transaction(
        user_id=current_user.id,
        amount=reward_amount,
        transaction_type="reward",
        description=f"Reward for incident #{incident_id}: {incident.title}",
        metadata={
            "incident_id": incident_id,
            "incident_title": incident.title,
            "severity": incident.severity,
            "confidence_score": incident.confidence_score,
            "reward_type": "incident_verification",
        }
    )
    
    db.add(transaction)
    
    # Update wallet
    wallet = await get_user_wallet(current_user.id, db)
    wallet.balance += reward_amount
    wallet.updated_at = datetime.utcnow()
    
    # Mark incident as rewarded
    incident.coins_awarded = True
    incident.coins_awarded_at = datetime.utcnow()
    incident.coins_amount = reward_amount
    
    db.commit()
    
    # Update user reputation
    current_user.reputation_score += 10  # Small reputation boost for verified incidents
    db.commit()
    
    return {
        "message": "Reward claimed successfully",
        "reward_amount": reward_amount,
        "new_balance": wallet.balance,
        "transaction_id": transaction.id,
        "incident_id": incident_id,
    }


@router.get("/stats")
async def get_reward_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get comprehensive reward statistics.
    """
    # User-specific stats
    user_wallet = await get_user_wallet(current_user.id, db)
    
    # Total rewards earned
    total_earned = db.query(func.sum(Transaction.amount)).filter(
        Transaction.user_id == current_user.id,
        Transaction.transaction_type == "reward",
    ).scalar() or 0
    
    # Total spent
    total_spent = db.query(func.sum(Transaction.amount)).filter(
        Transaction.user_id == current_user.id,
        Transaction.transaction_type == "purchase",
    ).scalar() or 0
    
    # Reward breakdown by type
    reward_types = db.query(
        Transaction.metadata["reward_type"].astext.label("reward_type"),
        func.count(Transaction.id).label("count"),
        func.sum(Transaction.amount).label("total_amount"),
    ).filter(
        Transaction.user_id == current_user.id,
        Transaction.transaction_type == "reward",
    ).group_by("reward_type").all()
    
    # Recent rewards (last 7 days)
    week_ago = datetime.utcnow() - timedelta(days=7)
    recent_rewards = db.query(func.sum(Transaction.amount)).filter(
        Transaction.user_id == current_user.id,
        Transaction.transaction_type == "reward",
        Transaction.created_at >= week_ago,
    ).scalar() or 0
    
    # User rank
    user_rank = None
    all_users = db.query(
        User.id,
        func.sum(Transaction.amount).label("total_rewards")
    ).join(
        Transaction, Transaction.user_id == User.id
    ).filter(
        Transaction.transaction_type == "reward",
        User.is_active == True,
    ).group_by(User.id).all()
    
    sorted_users = sorted(all_users, key=lambda x: x[1] or 0, reverse=True)
    for rank, (user_id, _) in enumerate(sorted_users, 1):
        if user_id == current_user.id:
            user_rank = rank
            break
    
    return {
        "user_id": current_user.id,
        "current_balance": user_wallet.balance,
        "total_earned": total_earned,
        "total_spent": total_spent,
        "net_earnings": total_earned - total_spent,
        "recent_earnings_7d": recent_rewards,
        "reward_breakdown": {
            rt.reward_type or "other": {
                "count": rt.count,
                "total_amount": rt.total_amount or 0
            }
            for rt in reward_types
        },
        "user_rank": user_rank,
        "total_users": len(sorted_users),
        "generated_at": datetime.utcnow().isoformat(),
    }


@router.get("/premium/features")
async def get_premium_features(
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get list of premium features and their costs.
    """
    premium_features = [
        {
            "id": "advanced_analytics",
            "name": "Advanced Analytics",
            "description": "Get detailed analytics and insights",
            "cost_coins": 100,
            "duration_days": 30,
            "category": "analytics",
        },
        {
            "id": "premium_briefings",
            "name": "Premium Briefings",
            "description": "Access to advanced briefing features",
            "cost_coins": 200,
            "duration_days": 30,
            "category": "briefing",
        },
        {
            "id": "priority_support",
            "name": "Priority Support",
            "description": "Get priority customer support",
            "cost_coins": 50,
            "duration_days": 30,
            "category": "support",
        },
        {
            "id": "ad_free",
            "name": "Ad-Free Experience",
            "description": "Remove all advertisements",
            "cost_coins": 150,
            "duration_days": 30,
            "category": "experience",
        },
        {
            "id": "early_access",
            "name": "Early Access",
            "description": "Early access to new features",
            "cost_coins": 75,
            "duration_days": 30,
            "category": "access",
        },
    ]
    
    user_wallet = None
    if current_user:
        user_wallet = await get_user_wallet(current_user.id, db)
    
    return {
        "premium_features": premium_features,
        "user_balance": user_wallet.balance if user_wallet else 0,
        "currency": "WB-Coins",
        "exchange_rate": 100,  # 100 coins = $1 (example)
    }


@router.post("/purchase/premium/{feature_id}")
async def purchase_premium_feature(
    feature_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Purchase a premium feature.
    """
    # Get feature details
    feature = await get_premium_feature_details(feature_id)
    if not feature:
        raise HTTPException(status_code=404, detail="Premium feature not found")
    
    # Get user wallet
    wallet = await get_user_wallet(current_user.id, db)
    
    # Check balance
    if wallet.balance < feature["cost_coins"]:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient balance. Need {feature['cost_coins']} coins, have {wallet.balance}"
        )
    
    # Create purchase transaction
    transaction = Transaction(
        user_id=current_user.id,
        amount=feature["cost_coins"],
        transaction_type="purchase",
        description=f"Purchase: {feature['name']}",
        metadata={
            "feature_id": feature_id,
            "feature_name": feature["name"],
            "duration_days": feature["duration_days"],
            "category": feature["category"],
        }
    )
    
    db.add(transaction)
    
    # Update wallet
    wallet.balance -= feature["cost_coins"]
    wallet.updated_at = datetime.utcnow()
    
    # Activate feature for user
    # This would typically involve updating user permissions or creating a subscription record
    # For now, we'll just log the purchase
    print(f"User {current_user.id} purchased {feature['name']} for {feature['cost_coins']} coins")
    
    db.commit()
    
    return {
        "message": f"Successfully purchased {feature['name']}",
        "feature": feature["name"],
        "cost": feature["cost_coins"],
        "new_balance": wallet.balance,
        "transaction_id": transaction.id,
        "expires_at": (datetime.utcnow() + timedelta(days=feature["duration_days"])).isoformat(),
    }


async def get_premium_feature_details(feature_id: str) -> Optional[Dict[str, Any]]:
    """
    Get details of a premium feature.
    """
    features = {
        "advanced_analytics": {
            "name": "Advanced Analytics",
            "description": "Get detailed analytics and insights",
            "cost_coins": 100,
            "duration_days": 30,
            "category": "analytics",
        },
        "premium_briefings": {
            "name": "Premium Briefings",
            "description": "Access to advanced briefing features",
            "cost_coins": 200,
            "duration_days": 30,
            "category": "briefing",
        },
        "priority_support": {
            "name": "Priority Support",
            "description": "Get priority customer support",
            "cost_coins": 50,
            "duration_days": 30,
            "category": "support",
        },
        "ad_free": {
            "name": "Ad-Free Experience",
            "description": "Remove all advertisements",
            "cost_coins": 150,
            "duration_days": 30,
            "category": "experience",
        },
        "early_access": {
            "name": "Early Access",
            "description": "Early access to new features",
            "cost_coins": 75,
            "duration_days": 30,
            "category": "access",
        },
    }
    
    return features.get(feature_id)


@router.get("/exchange/rate")
async def get_exchange_rate() -> Dict[str, Any]:
    """
    Get current exchange rate for WB-Coins.
    """
    return {
        "currency": "WB-Coins",
        "exchange_rates": {
            "USD": 0.01,  # 1 coin = $0.01
            "EUR": 0.009,
            "GBP": 0.008,
            "INR": 0.83,
        },
        "minimum_withdrawal": 1000,  # Minimum coins needed for withdrawal
        "last_updated": datetime.utcnow().isoformat(),
    }