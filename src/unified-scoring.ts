/**
 * Unified Scoring System
 * Matches the exact scoring logic from actions-bridge
 */

import { SMCWeights } from './types';
import { SMCAnalysis, SMCIndicators } from './smc-indicators';

export interface ConfluenceFactors {
  description: string;
  weight: number;
  value: any;
}

export interface UnifiedScore {
  score: number;
  breakdown: Record<string, number>;
  confluence: string[];
  bias: 'bullish' | 'bearish' | 'neutral';
  macroScore: number;
}

export class UnifiedScoring {
  /**
   * Calculate confluence score using exact actions-bridge logic
   */
  static calculateConfluence(
    analysis: SMCAnalysis,
    currentPrice: number,
    weights: SMCWeights
  ): UnifiedScore {
    const breakdown: Record<string, number> = {};
    const confluence: string[] = [];
    let totalScore = 0;
    
    // Determine overall bias
    let bias: 'bullish' | 'bearish' | 'neutral' = 'neutral';
    if (analysis.trend === 'up' && analysis.bos === 'up') {
      bias = 'bullish';
    } else if (analysis.trend === 'down' && analysis.bos === 'down') {
      bias = 'bearish';
    } else if (analysis.trend === 'up' || analysis.bos === 'up') {
      bias = 'bullish';
    } else if (analysis.trend === 'down' || analysis.bos === 'down') {
      bias = 'bearish';
    }
    
    // TREND + STRUCTURE ALIGNMENT (0-40 pts)
    if (analysis.trend && analysis.bos && analysis.trend === analysis.bos) {
      breakdown.trend_structure = weights.trend_structure;
      confluence.push(`Strong ${analysis.trend}trend (SMA + BOS ${analysis.bos})`);
    } else if (analysis.trend) {
      breakdown.trend_structure = weights.trend_structure * 0.5;
      confluence.push(`Trend ${analysis.trend}`);
    } else if (analysis.bos) {
      breakdown.trend_structure = weights.trend_structure * 0.3;
      confluence.push(`BOS ${analysis.bos}`);
    } else {
      breakdown.trend_structure = 0;
    }
    totalScore += breakdown.trend_structure || 0;
    
    // DIRECTIONAL ORDER BLOCKS (0-30 pts)
    const relevantOBs = analysis.orderBlocks.filter(ob => {
      const obHigh = Math.max(ob.open, ob.close);
      const obLow = Math.min(ob.open, ob.close);
      return currentPrice >= obLow * 0.95 && currentPrice <= obHigh * 1.05;
    });
    
    const bullOBs = relevantOBs.filter(ob => ob.type === 'bull');
    const bearOBs = relevantOBs.filter(ob => ob.type === 'bear');
    
    if (bias === 'bullish' && bullOBs.length > 0) {
      const nearOB = bullOBs.some((ob: any) => {
        const obHigh = Math.max(ob.open, ob.close);
        const obLow = Math.min(ob.open, ob.close);
        const dist = Math.abs(currentPrice - obLow) / currentPrice;
        return dist < 0.05;
      });
      breakdown.order_blocks = nearOB ? weights.order_blocks : weights.order_blocks * 0.6;
      confluence.push(nearOB ? `Bull OB nearby (${bullOBs.length})` : `Bull OB (${bullOBs.length})`);
    } else if (bias === 'bearish' && bearOBs.length > 0) {
      const nearOB = bearOBs.some((ob: any) => {
        const obHigh = Math.max(ob.open, ob.close);
        const dist = Math.abs(currentPrice - obHigh) / currentPrice;
        return dist < 0.05;
      });
      breakdown.order_blocks = nearOB ? weights.order_blocks : weights.order_blocks * 0.6;
      confluence.push(nearOB ? `Bear OB nearby (${bearOBs.length})` : `Bear OB (${bearOBs.length})`);
    } else if (relevantOBs.length > 0) {
      breakdown.order_blocks = weights.order_blocks * 0.3;
      confluence.push(`Mixed OBs (${bullOBs.length}B ${bearOBs.length}S)`);
    } else {
      breakdown.order_blocks = 0;
    }
    totalScore += breakdown.order_blocks || 0;
    
    // DIRECTIONAL FVGs (0-20 pts)
    const relevantFVGs = analysis.fvg.filter((fvg: any) => {
      return currentPrice >= fvg.from * 0.98 && currentPrice <= fvg.to * 1.02;
    });
    
    const bullFVGs = relevantFVGs.filter((fvg: any) => fvg.type === 'bull');
    const bearFVGs = relevantFVGs.filter((fvg: any) => fvg.type === 'bear');
    
    if (bias === 'bullish' && bullFVGs.length > 0) {
      const nearFVG = bullFVGs.some((fvg: any) => {
        const dist = Math.abs(currentPrice - fvg.from) / currentPrice;
        return dist < 0.03;
      });
      breakdown.fvgs = nearFVG ? weights.fvgs : weights.fvgs * 0.5;
      confluence.push(nearFVG ? `Bull FVG nearby (${bullFVGs.length})` : `Bull FVG (${bullFVGs.length})`);
    } else if (bias === 'bearish' && bearFVGs.length > 0) {
      const nearFVG = bearFVGs.some((fvg: any) => {
        const dist = Math.abs(currentPrice - fvg.to) / currentPrice;
        return dist < 0.03;
      });
      breakdown.fvgs = nearFVG ? weights.fvgs : weights.fvgs * 0.5;
      confluence.push(nearFVG ? `Bear FVG nearby (${bearFVGs.length})` : `Bear FVG (${bearFVGs.length})`);
    } else if (relevantFVGs.length > 0) {
      breakdown.fvgs = weights.fvgs;
      confluence.push(`${relevantFVGs.length} FVG(s)`);
    } else {
      breakdown.fvgs = 0;
    }
    totalScore += breakdown.fvgs || 0;
    
    // EMA ALIGNMENT (0-15 pts)
    if (analysis.ema50 && analysis.ema200) {
      const emaTrend = analysis.ema50 > analysis.ema200 ? 'bullish' : 'bearish';
      if ((bias === 'bullish' && emaTrend === 'bullish') || 
          (bias === 'bearish' && emaTrend === 'bearish')) {
        breakdown.ema_alignment = weights.ema_alignment;
        confluence.push(`EMA aligned ${emaTrend}`);
      } else {
        breakdown.ema_alignment = -5;
        confluence.push(`EMA divergent (${emaTrend})`);
      }
    } else {
      breakdown.ema_alignment = 0;
    }
    totalScore += breakdown.ema_alignment || 0;
    
    // LIQUIDITY LEVELS (0-10 pts)
    const hasLiquidity = analysis.liquidityZones.highs.length > 0 || 
                         analysis.liquidityZones.lows.length > 0;
    if (hasLiquidity) {
      breakdown.liquidity = weights.liquidity;
      confluence.push(`Liquidity zones: ${analysis.liquidityZones.highs.length}H ${analysis.liquidityZones.lows.length}L`);
    } else {
      breakdown.liquidity = 0;
    }
    totalScore += breakdown.liquidity || 0;
    
    // MTF BONUS (0-35 pts) - Added separately
    breakdown.mtf_bonus = weights.mtf_bonus;
    totalScore += breakdown.mtf_bonus;
    
    // RSI PENALTIES/BONUSES
    if (analysis.rsi) {
      if (analysis.rsi > 70) {
        breakdown.rsi_penalty = weights.rsi_penalty;
        confluence.push(`RSI overbought (${analysis.rsi.toFixed(1)})`);
      } else if (analysis.rsi < 30) {
        if (bias === 'bullish') {
          breakdown.rsi_penalty = weights.rsi_penalty * 0.5;
          confluence.push(`RSI oversold bounce setup (${analysis.rsi.toFixed(1)})`);
        } else {
          breakdown.rsi_penalty = weights.rsi_penalty;
          confluence.push(`RSI oversold (${analysis.rsi.toFixed(1)})`);
        }
      } else if (analysis.rsi >= 40 && analysis.rsi <= 60) {
        breakdown.rsi_penalty = 5;
        confluence.push(`RSI neutral (${analysis.rsi.toFixed(1)})`);
      } else {
        breakdown.rsi_penalty = 0;
      }
    } else {
      breakdown.rsi_penalty = 0;
    }
    totalScore += breakdown.rsi_penalty || 0;
    
    return {
      score: Math.max(0, totalScore),
      breakdown,
      confluence,
      bias,
      macroScore: 0 // Will be set externally
    };
  }
  
  /**
   * Calculate MTF alignment bonus
   * Checks if higher and lower timeframes agree
   */
  static calculateMTFBonus(
    daily: SMCAnalysis,
    hourly: SMCAnalysis | null,
    fiveMin: SMCAnalysis | null
  ): { bonus: number; factors: string[] } {
    let bonus = 0;
    const factors: string[] = [];
    
    // Check if HTF (1d) and LTF (1h) trend alignment
    if (daily.trend && hourly && hourly.trend) {
      if (daily.trend === hourly.trend) {
        bonus += 20;
        factors.push(`HTF+LTF aligned (${daily.trend})`);
      } else {
        bonus -= 10;
        factors.push(`HTF/LTF divergent (${daily.trend}/${hourly.trend})`);
      }
    }
    
    // Check if all 3 timeframes align
    if (daily.trend && hourly && hourly.trend && fiveMin && fiveMin.trend) {
      if (daily.trend === hourly.trend && hourly.trend === fiveMin.trend) {
        bonus += 15; // Triple timeframe alignment
        factors.push(`Triple TF alignment (${daily.trend})`);
      }
    }
    
    return {
      bonus: Math.max(-15, bonus),
      factors
    };
  }
}