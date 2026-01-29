/**
 * Trade Feature Extractor
 * Extracts detailed features from each trade setup for ML training
 */

import { Candle } from './smc-indicators.js';
import { SMCAnalysis, SMCIndicators } from './smc-indicators.js';
import fs from 'fs';

// Trade outcome data (for adding to features after backtest)
export interface BacktestTrade {
  outcome: 'WIN' | 'LOSS';
  pnl: number;
  pnl_percent: number;
  exit_reason: string;
  holding_periods: number;
}
import path from 'path';

// Load configuration
let featureConfig: any;
try {
  const configPath = path.join(process.cwd(), 'config', 'features.json');
  featureConfig = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
} catch (error) {
  console.warn('Warning: Could not load features.json config, using defaults');
  featureConfig = {
    feature_extraction: {
      lookback_periods: { default: 200 },
      thresholds: {
        ob_near_percent: 5,
        ob_distance_max: 1,
        fvg_near_percent: 2,
        fvg_distance_max: 1,
        fvg_size_max: 0.5,
        ob_age_max: 100,
        rsi_overbought: 70,
        rsi_oversold: 30,
        mtf_trend_threshold: 0.5
      }
    }
  };
}

export interface TradeFeatures {
  // Entry conditions
  entry_price: number;
  entry_time: number;
  direction: 'long' | 'short';
  
  // Trend features
  trend_direction: 'up' | 'down' | 'neutral';
  trend_strength: number;  // 0-1, based on SMA slope
  trend_bos_aligned: boolean;  // Does BOS match trend?
  
  // Order Block features
  ob_near: boolean;
  ob_distance: number;  // % away from price
  ob_type: 'bull' | 'bear' | 'none';
  ob_size: number;  // % of candle body
  ob_age: number;  // How many candles ago (0-100+)
  
  // FVG features
  fvg_near: boolean;
  fvg_count: number;  // Total relevant FVGs
  fvg_nearest_distance: number;  // % away
  fvg_size: number;  // Average gap size %
  fvg_type: 'bull' | 'bear' | 'mixed';
  
  // EMA alignment
  ema_aligned: boolean;
  ema_trend: 'up' | 'down' | 'neutral';
  
  // RSI state
  rsi_value: number;
  rsi_state: 'overbought' | 'oversold' | 'neutral';
  
  // Liquidity
  liquidity_near: boolean;
  liquidity_count: number;
  
  // Multi-timeframe
  mtf_aligned: boolean;
  
  // Market state
  volatility: number;  // ATR % of price
  atr_value: number;
  
  // Price position
  price_position: number;  // 0-1, where in recent range
  distance_to_high: number;  // % from recent high
  distance_to_low: number;  // % from recent low
  
  // Volume features
  volume_spike: boolean;  // Is volume 2x+ average?
  volume_ratio: number;  // Current / avg volume
  
  // Confluence
  confluence_score: number;  // 0-1, how many signals align
  confluence_count: number;  // Number of confluence factors
  
  // Market session
  session: 'asian' | 'london' | 'newyork' | 'overlap' | 'off-hours';
  
  // Psychological
  days_since_loss: number;  // Days since last loss (0-100+)
  streak_type: 'win' | 'loss' | 'none';
  
  // Risk/Reward
  potential_rr: number;  // Potential risk/reward ratio
  
  // Outcome (for training)
  outcome: 'WIN' | 'LOSS';
  pnl: number;
  pnl_percent: number;
  exit_reason: string;
  holding_periods: number;
}

export class FeatureExtractor {
  /**
   * Extract features from a trade entry point
   */
  static extractFeatures(
    candles: Candle[],
    index: number,
    analysis: SMCAnalysis,
    score: number,
    direction: 'long' | 'short'
  ): Omit<TradeFeatures, 'outcome' | 'pnl' | 'pnl_percent' | 'exit_reason' | 'holding_periods'> {
    const currentCandle = candles[index];
    const config = featureConfig.feature_extraction;
    const lookback = config.lookback_periods.default;
    const historicalCandles = candles.slice(Math.max(0, index - lookback), index + 1);
    
    // Trend features
    const trend_direction = analysis.trend || 'neutral';
    const trend_strength = this.calculateTrendStrength(historicalCandles);
    const trend_bos_aligned = analysis.bos === analysis.trend;
    
    // Order Block features
    const relevantOBs = analysis.orderBlocks.filter(ob => {
      const obHigh = Math.max(ob.open, ob.close);
      const obLow = Math.min(ob.open, ob.close);
      const nearPercent = 1 + (config.thresholds.ob_near_percent / 100);
      return currentCandle.close >= obLow * (2 - nearPercent) && currentCandle.close <= obHigh * nearPercent;
    });
    
    const bullOBs = relevantOBs.filter(ob => ob.type === 'bull');
    const bearOBs = relevantOBs.filter(ob => ob.type === 'bear');
    const matchingOBs = direction === 'long' ? bullOBs : bearOBs;
    
    const ob_near = matchingOBs.length > 0;
    let ob_distance = 999;
    let ob_size = 0;
    let ob_age = 0;
    let ob_type: 'bull' | 'bear' | 'none' = 'none';
    
    if (matchingOBs.length > 0) {
      const nearestOB = matchingOBs.reduce((nearest, ob) => {
        const dist = direction === 'long'
          ? Math.abs(currentCandle.close - Math.min(ob.open, ob.close))
          : Math.abs(currentCandle.close - Math.max(ob.open, ob.close));
        const nearestDist = direction === 'long'
          ? Math.abs(currentCandle.close - Math.min(nearest.open, nearest.close))
          : Math.abs(currentCandle.close - Math.max(nearest.open, nearest.close));
        return dist < nearestDist ? ob : nearest;
      });
      
      const obHigh = Math.max(nearestOB.open, nearestOB.close);
      const obLow = Math.min(nearestOB.open, nearestOB.close);
      ob_distance = direction === 'long'
        ? Math.abs(currentCandle.close - obLow) / currentCandle.close
        : Math.abs(currentCandle.close - obHigh) / currentCandle.close;
      
      ob_size = Math.abs(nearestOB.close - nearestOB.open) / nearestOB.open;
      ob_age = index - nearestOB.index;
      ob_type = nearestOB.type;
    }
    
    // FVG features
    const relevantFVGs = analysis.fvg.filter((fvg: any) => {
      return currentCandle.close >= fvg.from * 0.98 && currentCandle.close <= fvg.to * 1.02;
    });
    
    const bullFVGs = relevantFVGs.filter((fvg: any) => fvg.type === 'bull');
    const bearFVGs = relevantFVGs.filter((fvg: any) => fvg.type === 'bear');
    const matchingFVGs = direction === 'long' ? bullFVGs : bearFVGs;
    
    const fvg_near = relevantFVGs.length > 0;
    const fvg_count = relevantFVGs.length;
    
    let fvg_nearest_distance = 999;
    let fvg_size = 0;
    let fvg_type: 'bull' | 'bear' | 'mixed' = 'mixed';
    
    if (matchingFVGs.length > 0) {
      const nearestFVG = matchingFVGs[0];
      fvg_nearest_distance = Math.abs(currentCandle.close - nearestFVG.from) / currentCandle.close;
      fvg_size = (nearestFVG.to - nearestFVG.from) / nearestFVG.from;
      fvg_type = direction === 'long' ? 'bull' : 'bear';
    }
    
    // EMA alignment
    const ema_aligned: boolean = !!(analysis.ema50 && analysis.ema200 && 
      ((direction === 'long' && analysis.ema50 > analysis.ema200) ||
       (direction === 'short' && analysis.ema50 < analysis.ema200)));
    const ema_trend = analysis.ema50 && analysis.ema200 
      ? (analysis.ema50 > analysis.ema200 ? 'up' : 'down')
      : 'neutral';
    
    // RSI state
    const rsi_value = analysis.rsi || 50;
    let rsi_state: 'overbought' | 'oversold' | 'neutral' = 'neutral';
    if (rsi_value > 70) rsi_state = 'overbought';
    else if (rsi_value < 30) rsi_state = 'oversold';
    
    // Liquidity
    const liquidity_near = analysis.liquidityZones.highs.length > 0 || 
                         analysis.liquidityZones.lows.length > 0;
    const liquidity_count = analysis.liquidityZones.highs.length + 
                         analysis.liquidityZones.lows.length;
    
    // Multi-timeframe alignment check
    const mtf_aligned: boolean = this.checkMTFAlignment(
      trend_direction,
      direction,
      currentCandle,
      historicalCandles
    );
    
    // Market state
    const atr = analysis.atr || (currentCandle.high - currentCandle.low);
    const volatility = atr / currentCandle.close;
    
    // Price position (where in recent range)
    const recentHigh = Math.max(...historicalCandles.slice(-50).map(c => c.high));
    const recentLow = Math.min(...historicalCandles.slice(-50).map(c => c.low));
    const range = recentHigh - recentLow;
    const price_position = range > 0 ? (currentCandle.close - recentLow) / range : 0.5;
    const distance_to_high = range > 0 ? (recentHigh - currentCandle.close) / currentCandle.close : 0;
    const distance_to_low = range > 0 ? (currentCandle.close - recentLow) / currentCandle.close : 0;
    
    // Volume features
    const recentVolumes = historicalCandles.slice(-20).map(c => c.volume);
    const avgVolume = recentVolumes.reduce((sum, v) => sum + v, 0) / recentVolumes.length;
    const volume_ratio = avgVolume > 0 ? currentCandle.volume / avgVolume : 1;
    const volume_spike = volume_ratio >= 2;
    
    // Confluence score
    const confluenceFactors = [
      ob_near,
      fvg_near,
      ema_aligned,
      liquidity_near,
      mtf_aligned,
      trend_bos_aligned
    ].filter(Boolean);
    const confluence_count = confluenceFactors.length;
    const confluence_score = Math.min(confluence_count / 6, 1);
    
    // Market session
    const hour = new Date(currentCandle.timestamp).getUTCHours();
    let session: 'asian' | 'london' | 'newyork' | 'overlap' | 'off-hours' = 'off-hours';
    if (hour >= 0 && hour < 6) session = 'asian';
    else if (hour >= 6 && hour < 8) session = 'london';
    else if (hour >= 8 && hour < 12) session = 'overlap';
    else if (hour >= 12 && hour < 16) session = 'newyork';
    else if (hour >= 16 && hour < 20) session = 'newyork';
    
    // Psychological (simplified - would need trade history)
    const days_since_loss = 0;  // Would track from trade journal
    const streak_type: 'win' | 'loss' | 'none' = 'none';
    
    // Risk/Reward potential (based on ATR and nearest resistance)
    const potential_rr = 2;  // Default 1:2 RR, would calculate from actual levels
    
    return {
      entry_price: currentCandle.close,
      entry_time: currentCandle.timestamp,
      direction,
      trend_direction,
      trend_strength,
      trend_bos_aligned,
      ob_near,
      ob_distance: Math.min(ob_distance, 1),
      ob_type,
      ob_size,
      ob_age: Math.min(ob_age, 100),
      fvg_near,
      fvg_count,
      fvg_nearest_distance: Math.min(fvg_nearest_distance, 1),
      fvg_size: Math.min(fvg_size, 0.5),
      fvg_type,
      ema_aligned,
      ema_trend,
      rsi_value,
      rsi_state,
      liquidity_near,
      liquidity_count,
      mtf_aligned,
      volatility,
      atr_value: atr,
      price_position,
      distance_to_high,
      distance_to_low,
      volume_spike,
      volume_ratio,
      confluence_score,
      confluence_count,
      session,
      days_since_loss,
      streak_type,
      potential_rr
    };
  }
  
  /**
   * Calculate trend strength from SMA slope
   */
  private static calculateTrendStrength(candles: Candle[]): number {
    if (candles.length < 20) return 0;
    
    // Calculate 20-period SMA
    const period = 20;
    const sma: number[] = [];
    for (let i = period - 1; i < candles.length; i++) {
      let sum = 0;
      for (let j = 0; j < period; j++) {
        sum += candles[i - j].close;
      }
      sma.push(sum / period);
    }
    
    if (sma.length < 5) return 0;
    
    // Calculate slope of last 5 SMAs
    const slope = (sma[sma.length - 1] - sma[sma.length - 5]) / 5;
    const avgPrice = candles[candles.length - 1].close;
    
    // Normalize to 0-1
    const normalizedSlope = Math.abs(slope / avgPrice) * 100;
    return Math.min(Math.max(normalizedSlope, 0), 1);
  }
  
  /**
   * Add outcome to features (after trade completes)
   */
  static addOutcome(
    features: Omit<TradeFeatures, 'outcome' | 'pnl' | 'pnl_percent' | 'exit_reason' | 'holding_periods'>,
    trade: BacktestTrade
  ): TradeFeatures {
    return {
      ...features,
      outcome: trade.outcome,
      pnl: trade.pnl,
      pnl_percent: trade.pnl_percent,
      exit_reason: trade.exit_reason,
      holding_periods: trade.holding_periods
    };
  }
  
  /**
   * Check multi-timeframe alignment
   * Simulates checking if higher timeframes agree with current trend
   */
  private static checkMTFAlignment(
    trend_direction: 'up' | 'down' | 'neutral',
    direction: 'long' | 'short',
    currentCandle: Candle,
    historicalCandles: Candle[]
  ): boolean {
    if (historicalCandles.length < 50) return true; // Not enough data
    
    // Get trend on different lookback periods (simulating MTF)
    const shortTermTrend = this.getTrendForPeriod(historicalCandles, 20);
    const mediumTermTrend = this.getTrendForPeriod(historicalCandles, 50);
    const longTermTrend = this.getTrendForPeriod(historicalCandles, 100);
    
    // Check alignment
    const isLong = direction === 'long';
    
    // For long trades: want higher timeframes to show upward trend
    if (isLong) {
      const shortAligned = shortTermTrend === 'up' || shortTermTrend === 'neutral';
      const mediumAligned = mediumTermTrend === 'up' || mediumTermTrend === 'neutral';
      const longAligned = longTermTrend === 'up' || longTermTrend === 'neutral';
      
      // At least 2 out of 3 should be aligned
      return [shortAligned, mediumAligned, longAligned].filter(Boolean).length >= 2;
    } 
    // For short trades: want higher timeframes to show downward trend
    else {
      const shortAligned = shortTermTrend === 'down' || shortTermTrend === 'neutral';
      const mediumAligned = mediumTermTrend === 'down' || mediumTermTrend === 'neutral';
      const longAligned = longTermTrend === 'down' || longTermTrend === 'neutral';
      
      // At least 2 out of 3 should be aligned
      return [shortAligned, mediumAligned, longAligned].filter(Boolean).length >= 2;
    }
  }
  
  /**
   * Get trend direction for a specific period
   */
  private static getTrendForPeriod(candles: Candle[], period: number): 'up' | 'down' | 'neutral' {
    if (candles.length < period) return 'neutral';
    
    const recentCandles = candles.slice(-period);
    const firstPrice = recentCandles[0].close;
    const lastPrice = recentCandles[recentCandles.length - 1].close;
    
    const percentChange = (lastPrice - firstPrice) / firstPrice * 100;
    
    if (percentChange > 0.5) return 'up';
    if (percentChange < -0.5) return 'down';
    return 'neutral';
  }
}
