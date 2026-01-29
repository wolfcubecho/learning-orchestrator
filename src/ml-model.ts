/**
 * Machine Learning Model for Trading
 * Learns patterns from historical trades to predict win probability
 */

import { TradeFeatures } from './trade-features.js';

export interface FeatureBin {
  min: number;
  max: number;
  count: number;
  wins: number;
  winRate: number;
  value?: any;  // Categorical value (for matching)
}

export interface ModelFeature {
  name: string;
  bins: FeatureBin[];
  importance: number;  // 0-1, how predictive this feature is
}

export interface Prediction {
  winProbability: number;  // 0-1
  confidence: number;  // 0-1, based on sample size
  keyFeatures: string[];  // Which features are driving the prediction
  reason: string;  // Human-readable explanation
}

export class TradingMLModel {
  private features: Map<string, ModelFeature> = new Map();
  private minSampleSize = 30;  // Minimum samples to make prediction
  private trained = false;
  
  /**
   * Train the model on historical trades
   */
  train(trades: TradeFeatures[]): void {
    console.log(`\n=== Training ML Model ===`);
    console.log(`Training on ${trades.length} trades...`);
    
    this.features.clear();
    
    // Extract and bin each feature
    this.binFeature('trend_strength', trades, 10);
    this.binFeature('ob_distance', trades, 10);
    this.binFeature('ob_size', trades, 10);
    this.binFeature('ob_age', trades, 10);
    this.binFeature('fvg_nearest_distance', trades, 10);
    this.binFeature('fvg_size', trades, 10);
    this.binFeature('fvg_count', trades, 10);
    this.binFeature('volatility', trades, 10);
    this.binFeature('rsi_value', trades, 10);
    this.binFeature('atr_value', trades, 10);
    
    // New features
    this.binFeature('price_position', trades, 10);
    this.binFeature('distance_to_high', trades, 10);
    this.binFeature('distance_to_low', trades, 10);
    this.binFeature('volume_ratio', trades, 10);
    this.binFeature('confluence_score', trades, 10);
    this.binFeature('potential_rr', trades, 10);
    
    // Categorical features
    this.binCategorical('trend_direction', trades);
    this.binCategorical('trend_bos_aligned', trades);
    this.binCategorical('ob_near', trades);
    this.binCategorical('ob_type', trades);
    this.binCategorical('fvg_near', trades);
    this.binCategorical('fvg_type', trades);
    this.binCategorical('ema_aligned', trades);
    this.binCategorical('ema_trend', trades);
    this.binCategorical('rsi_state', trades);
    this.binCategorical('liquidity_near', trades);
    this.binCategorical('mtf_aligned', trades);
    this.binCategorical('direction', trades);
    this.binCategorical('volume_spike', trades);
    this.binCategorical('session', trades);
    this.binCategorical('days_since_loss', trades);
    this.binCategorical('streak_type', trades);
    
    // Calculate feature importance
    this.calculateFeatureImportance();
    
    this.trained = true;
    
    // Print insights
    this.printInsights();
  }
  
  /**
   * Bin numerical features into ranges
   */
  private binFeature(featureName: string, trades: TradeFeatures[], numBins: number): void {
    const values = trades.map(t => t[featureName as keyof TradeFeatures] as number);
    const minVal = Math.min(...values);
    const maxVal = Math.max(...values);
    const binSize = (maxVal - minVal) / numBins;
    
    const bins: FeatureBin[] = [];
    
    for (let i = 0; i < numBins; i++) {
      const binMin = minVal + (i * binSize);
      const binMax = minVal + ((i + 1) * binSize);
      
      const tradesInBin = trades.filter(t => {
        const val = t[featureName as keyof TradeFeatures] as number;
        return val >= binMin && val < binMax;
      });
      
      const wins = tradesInBin.filter(t => t.outcome === 'WIN').length;
      
      bins.push({
        min: binMin,
        max: binMax,
        count: tradesInBin.length,
        wins,
        winRate: tradesInBin.length > 0 ? wins / tradesInBin.length : 0
      });
    }
    
    this.features.set(featureName, {
      name: featureName,
      bins,
      importance: 0  // Will calculate later
    });
  }
  
  /**
   * Bin categorical features
   */
  private binCategorical(featureName: string, trades: TradeFeatures[]): void {
    const uniqueValues = new Set(trades.map(t => t[featureName as keyof TradeFeatures]));
    
    const bins: FeatureBin[] = [];
    
    for (const value of uniqueValues) {
      const tradesWithValue = trades.filter(t => 
        t[featureName as keyof TradeFeatures] === value
      );
      const wins = tradesWithValue.filter(t => t.outcome === 'WIN').length;
      
      bins.push({
        min: 0,  // Not used for categorical
        max: 0,
        count: tradesWithValue.length,
        wins,
        winRate: tradesWithValue.length > 0 ? wins / tradesWithValue.length : 0,
        value  // Store categorical value for matching
      });
    }
    
    this.features.set(featureName, {
      name: featureName,
      bins,
      importance: 0
    });
  }
  
  /**
   * Calculate feature importance based on variance in win rates
   */
  private calculateFeatureImportance(): void {
    for (const [name, feature] of this.features.entries()) {
      const winRates = feature.bins.filter(b => b.count >= this.minSampleSize).map(b => b.winRate);
      
      if (winRates.length < 2) {
        feature.importance = 0;
        continue;
      }
      
      // Calculate variance in win rates
      const mean = winRates.reduce((sum, r) => sum + r, 0) / winRates.length;
      const variance = winRates.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / winRates.length;
      
      // Higher variance = more predictive
      feature.importance = Math.min(variance * 10, 1);
    }
  }
  
  /**
   * Predict win probability for a trade setup
   */
  predict(features: TradeFeatures): Prediction {
    if (!this.trained) {
      throw new Error('Model not trained. Call train() first.');
    }
    
    const weightedScores: { feature: string; score: number; winRate: number }[] = [];
    
    // Score each feature
    for (const [name, modelFeature] of this.features.entries()) {
      const featureValue = features[name as keyof TradeFeatures];
      
      // Find the bin for this value
      let binWinRate = 0.5;
      let binCount = 0;
      
      for (const bin of modelFeature.bins) {
        if (typeof featureValue === 'number') {
          if (featureValue >= bin.min && featureValue < bin.max) {
            binWinRate = bin.winRate;
            binCount = bin.count;
            break;
          }
        } else {
          // Categorical match - compare actual values
          if (bin.count > 0 && bin.value === featureValue) {
            binWinRate = bin.winRate;
            binCount = bin.count;
            break;
          }
        }
      }
      
      // Weight by feature importance and sample size
      const sampleWeight = Math.min(binCount / this.minSampleSize, 1);
      const score = binWinRate * modelFeature.importance * sampleWeight;
      
      weightedScores.push({
        feature: name,
        score,
        winRate: binWinRate
      });
    }
    
    // Sort by score
    weightedScores.sort((a, b) => b.score - a.score);
    
    // Calculate overall prediction (weighted average of top features)
    const topFeatures = weightedScores.slice(0, 10);
    const totalWeight = topFeatures.reduce((sum, f) => sum + f.score, 0);
    const weightedSum = topFeatures.reduce((sum, f) => sum + (f.score * f.winRate), 0);
    const winProbability = totalWeight > 0 ? weightedSum / totalWeight : 0.5;
    
    // Confidence based on total sample size of top features
    const totalSamples = topFeatures.reduce((sum, f) => sum + Math.min(f.score / f.winRate, 100), 0);
    const confidence = Math.min(totalSamples / 1000, 1);
    
    // Generate explanation
    const keyFeatures = topFeatures.slice(0, 5).map(f => f.feature);
    const reasons = topFeatures.slice(0, 5).map(f => {
      const winRate = (f.winRate * 100).toFixed(0);
      return `${f.feature} ${(f.winRate * 100).toFixed(0)}% win rate`;
    });
    
    return {
      winProbability,
      confidence,
      keyFeatures,
      reason: reasons.join(', ')
    };
  }
  
  /**
   * Print model insights
   */
  private printInsights(): void {
    console.log(`\n📊 Feature Importance Analysis:`);
    
    const sortedFeatures = Array.from(this.features.entries())
      .sort((a, b) => b[1].importance - a[1].importance)
      .slice(0, 10);
    
    for (const [name, feature] of sortedFeatures) {
      const bestBin = feature.bins
        .filter(b => b.count >= this.minSampleSize)
        .sort((a, b) => b.winRate - a.winRate)[0];
      
      if (bestBin) {
        console.log(`  ${name.padEnd(25)} ${(feature.importance * 100).toFixed(0)}% importance`);
        console.log(`    Best range: ${this.formatRange(name, bestBin)} = ${(bestBin.winRate * 100).toFixed(0)}% win rate (${bestBin.count} trades)`);
      }
    }
    
    console.log(`\n💡 High-Probability Setup Characteristics:`);
    
    // Find characteristics of best performing trades
    const bestTrades = this.findBestPerformingBins();
    
    for (const insight of bestTrades.slice(0, 5)) {
      console.log(`  ${insight}`);
    }
  }
  
  /**
   * Find best performing ranges
   */
  private findBestPerformingBins(): string[] {
    const insights: string[] = [];
    
    for (const [name, feature] of this.features.entries()) {
      const bestBins = feature.bins
        .filter(b => b.count >= this.minSampleSize)
        .sort((a, b) => b.winRate - a.winRate)
        .slice(0, 2);
      
      for (const bin of bestBins) {
        if (bin.winRate > 0.55) {
          insights.push(
            `${name} ${this.formatRange(name, bin)} = ${(bin.winRate * 100).toFixed(0)}% win rate`
          );
        }
      }
    }
    
    return insights;
  }
  
  /**
   * Format bin range for display
   */
  private formatRange(featureName: string, bin: FeatureBin): string {
    if (bin.max === 0 && bin.min === 0) {
      return '(categorical)';
    }
    
    const val = bin.min;
    switch (featureName) {
      case 'trend_strength':
        return `${(val * 100).toFixed(1)}%`;
      case 'ob_distance':
      case 'fvg_nearest_distance':
        return `${(val * 100).toFixed(2)}%`;
      case 'ob_size':
      case 'fvg_size':
        return `${(val * 100).toFixed(2)}%`;
      case 'ob_age':
        return `${Math.floor(val)} candles`;
      case 'volatility':
        return `${(val * 100).toFixed(2)}%`;
      case 'rsi_value':
        return `${val.toFixed(0)}`;
      case 'atr_value':
        return `$${val.toFixed(2)}`;
      default:
        return `${val.toFixed(3)}`;
    }
  }
  
  /**
   * Get model statistics
   */
  getStats(): any {
    return {
      trained: this.trained,
      numFeatures: this.features.size,
      minSampleSize: this.minSampleSize
    };
  }
}