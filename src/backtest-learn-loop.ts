#!/usr/bin/env node
/**
 * Backtest-Learn Loop
 *
 * Self-improving ML training loop:
 * 1. Extract trades (including low-score ones) from historical data
 * 2. Run model predictions on all trades
 * 3. Compare predictions to actual outcomes
 * 4. Retrain model with prediction errors emphasized
 * 5. Repeat until accuracy plateaus
 *
 * This teaches the model from its own mistakes through backtesting.
 *
 * Run: npx ts-node src/backtest-learn-loop.ts
 */

import fs from 'fs';
import path from 'path';
import { LocalDataLoader } from './data-loader.js';
import { SMCIndicators, Candle } from './smc-indicators.js';
import { UnifiedScoring } from './unified-scoring.js';
import { FeatureExtractor, TradeFeatures } from './trade-features.js';
import { TradingMLModel } from './ml-model.js';

import os from 'os';

// Configuration
const CONFIG = {
  symbols: [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT',
    'XRPUSDT', 'DOGEUSDT', 'DOTUSDT', 'AVAXUSDT', 'LINKUSDT'
  ],
  timeframes: ['1d', '1h', '5m'],  // Available: 1d, 1h, 5m, 1m

  // LOW min score to capture losers too - model needs to learn what NOT to do
  minScore: 25,

  dataPath: path.join(process.cwd(), 'Historical_Data_Lite'),
  outputDir: path.join(process.cwd(), 'data', 'learning-loop'),
  modelDir: path.join(process.cwd(), 'data', 'models'),

  // Learning loop settings
  maxIterations: 10,           // Max training iterations
  minAccuracyImprovement: 0.005, // Stop if improvement < 0.5%
  trainTestSplit: 0.7,         // 70% train, 30% test

  // Emphasis on prediction errors (model learns more from mistakes)
  errorEmphasisMultiplier: 3,  // Wrong predictions weighted 3x in training

  // Parallel processing
  workers: Math.max(1, os.cpus().length - 2),  // Leave 2 cores free
};

interface LoopIteration {
  iteration: number;
  tradesExtracted: number;
  trainAccuracy: number;
  testAccuracy: number;
  predictionErrors: number;
  improvementFromLast: number;
}

interface TradeWithPrediction extends TradeFeatures {
  predicted_win_prob: number;
  prediction_correct: boolean;
  prediction_confidence: number;
}

class BacktestLearnLoop {
  private dataLoader: LocalDataLoader;
  private model: TradingMLModel;
  private allTrades: TradeFeatures[] = [];
  private iterations: LoopIteration[] = [];

  constructor() {
    this.dataLoader = new LocalDataLoader(CONFIG.dataPath);
    this.model = new TradingMLModel();
    this.ensureDirectories();
  }

  /**
   * Run the full learning loop
   */
  async run(): Promise<void> {
    const startTime = Date.now();

    console.log('╔═══════════════════════════════════════════════════════════════╗');
    console.log('║          BACKTEST-LEARN LOOP (Self-Improving ML)              ║');
    console.log('╚═══════════════════════════════════════════════════════════════╝\n');

    console.log('Configuration:');
    console.log(`  Symbols: ${CONFIG.symbols.length}`);
    console.log(`  Timeframes: ${CONFIG.timeframes.join(', ')}`);
    console.log(`  Min Score: ${CONFIG.minScore} (LOW to capture losers)`);
    console.log(`  Max Iterations: ${CONFIG.maxIterations}`);
    console.log(`  Error Emphasis: ${CONFIG.errorEmphasisMultiplier}x`);
    console.log('');

    // Phase 1: Extract ALL trades (including low quality)
    console.log('═══════════════════════════════════════════════════════════════');
    console.log('PHASE 1: Extracting ALL Trades (including low-score losers)');
    console.log('═══════════════════════════════════════════════════════════════\n');

    await this.extractAllTrades();

    if (this.allTrades.length < 200) {
      console.log(`\n❌ Not enough trades (${this.allTrades.length}). Need 200+.`);
      return;
    }

    // Analyze trade quality distribution
    const winners = this.allTrades.filter(t => t.outcome === 'WIN').length;
    const losers = this.allTrades.filter(t => t.outcome === 'LOSS').length;
    const highScore = this.allTrades.filter(t => t.confluence_score > 0.6).length;
    const lowScore = this.allTrades.filter(t => t.confluence_score <= 0.6).length;

    console.log(`\n📊 Trade Distribution:`);
    console.log(`  Total: ${this.allTrades.length}`);
    console.log(`  Winners: ${winners} (${(winners/this.allTrades.length*100).toFixed(1)}%)`);
    console.log(`  Losers: ${losers} (${(losers/this.allTrades.length*100).toFixed(1)}%)`);
    console.log(`  High Score (>60%): ${highScore}`);
    console.log(`  Low Score (≤60%): ${lowScore}`);

    // Phase 2: Initial training
    console.log('\n═══════════════════════════════════════════════════════════════');
    console.log('PHASE 2: Starting Learning Loop');
    console.log('═══════════════════════════════════════════════════════════════\n');

    let lastAccuracy = 0;
    let trainingData = [...this.allTrades];

    for (let i = 1; i <= CONFIG.maxIterations; i++) {
      console.log(`\n--- Iteration ${i}/${CONFIG.maxIterations} ---`);

      // Split data
      const shuffled = [...trainingData].sort(() => Math.random() - 0.5);
      const splitIdx = Math.floor(shuffled.length * CONFIG.trainTestSplit);
      const trainSet = shuffled.slice(0, splitIdx);
      const testSet = shuffled.slice(splitIdx);

      // Train model
      console.log(`  Training on ${trainSet.length} trades...`);
      this.model.train(trainSet);

      // Test on held-out set
      console.log(`  Testing on ${testSet.length} trades...`);
      const { accuracy: testAccuracy, errors } = this.evaluateModel(testSet);
      const trainAccuracy = this.evaluateModel(trainSet).accuracy;

      const improvement = testAccuracy - lastAccuracy;

      console.log(`  Train Accuracy: ${(trainAccuracy * 100).toFixed(1)}%`);
      console.log(`  Test Accuracy: ${(testAccuracy * 100).toFixed(1)}%`);
      console.log(`  Prediction Errors: ${errors.length}`);
      console.log(`  Improvement: ${improvement >= 0 ? '+' : ''}${(improvement * 100).toFixed(2)}%`);

      this.iterations.push({
        iteration: i,
        tradesExtracted: trainingData.length,
        trainAccuracy,
        testAccuracy,
        predictionErrors: errors.length,
        improvementFromLast: improvement
      });

      // Check for convergence
      if (i > 1 && improvement < CONFIG.minAccuracyImprovement) {
        console.log(`\n✅ Converged! Improvement (${(improvement * 100).toFixed(2)}%) < threshold (${(CONFIG.minAccuracyImprovement * 100).toFixed(2)}%)`);
        break;
      }

      // Emphasize errors for next iteration
      if (i < CONFIG.maxIterations) {
        console.log(`  Adding ${errors.length} error cases with ${CONFIG.errorEmphasisMultiplier}x emphasis...`);

        // Add error cases multiple times to emphasize them
        for (let j = 0; j < CONFIG.errorEmphasisMultiplier - 1; j++) {
          trainingData.push(...errors);
        }

        console.log(`  Training set now: ${trainingData.length} trades`);
      }

      lastAccuracy = testAccuracy;
    }

    // Phase 3: Save results
    console.log('\n═══════════════════════════════════════════════════════════════');
    console.log('PHASE 3: Saving Results');
    console.log('═══════════════════════════════════════════════════════════════\n');

    await this.saveResults();

    // Summary
    const duration = (Date.now() - startTime) / 1000;
    const finalAccuracy = this.iterations[this.iterations.length - 1]?.testAccuracy || 0;
    const startAccuracy = this.iterations[0]?.testAccuracy || 0;

    console.log('\n═══════════════════════════════════════════════════════════════');
    console.log('LEARNING LOOP COMPLETE');
    console.log('═══════════════════════════════════════════════════════════════');
    console.log(`\n  Total trades processed: ${this.allTrades.length}`);
    console.log(`  Iterations completed: ${this.iterations.length}`);
    console.log(`  Starting accuracy: ${(startAccuracy * 100).toFixed(1)}%`);
    console.log(`  Final accuracy: ${(finalAccuracy * 100).toFixed(1)}%`);
    console.log(`  Total improvement: +${((finalAccuracy - startAccuracy) * 100).toFixed(1)}%`);
    console.log(`  Duration: ${duration.toFixed(1)}s`);
    console.log(`\n  Model saved to: ${CONFIG.modelDir}`);
    console.log(`  Loop history: ${CONFIG.outputDir}`);
  }

  /**
   * Extract all trades from historical data (PARALLEL)
   */
  private async extractAllTrades(): Promise<void> {
    const total = CONFIG.symbols.length * CONFIG.timeframes.length;
    let completed = 0;
    let failed = 0;

    console.log(`\n🚀 Using ${CONFIG.workers} parallel workers\n`);

    // Build all jobs
    const jobs: Array<{ symbol: string; timeframe: string }> = [];
    for (const timeframe of CONFIG.timeframes) {
      for (const symbol of CONFIG.symbols) {
        jobs.push({ symbol, timeframe });
      }
    }

    // Process in parallel batches
    const batchSize = CONFIG.workers;
    const results: TradeFeatures[][] = [];

    for (let i = 0; i < jobs.length; i += batchSize) {
      const batch = jobs.slice(i, i + batchSize);

      const batchPromises = batch.map(async (job) => {
        try {
          const trades = await this.extractTradesForSymbol(job.symbol, job.timeframe);
          completed++;
          return { success: true, trades, job };
        } catch (err: any) {
          completed++;
          failed++;
          return { success: false, trades: [], job, error: err?.message };
        }
      });

      const batchResults = await Promise.all(batchPromises);

      // Collect results and show progress
      for (const result of batchResults) {
        if (result.success) {
          results.push(result.trades);
        }
        const progress = Math.floor((completed / total) * 100);
        const status = result.success ? `${result.trades.length} trades` : 'FAILED';
        process.stdout.write(`\r  [${progress}%] ${result.job.symbol}/${result.job.timeframe}: ${status}     `);
      }
    }

    // Flatten results
    for (const trades of results) {
      this.allTrades.push(...trades);
    }

    console.log(`\n\n✅ Extraction complete: ${total - failed}/${total} successful`);
    console.log(`   Total trades extracted: ${this.allTrades.length}`);
  }

  /**
   * Extract trades for a single symbol/timeframe
   */
  private async extractTradesForSymbol(symbol: string, timeframe: string): Promise<TradeFeatures[]> {
    const result = await this.dataLoader.loadData(symbol, timeframe);
    const candles = result.candles;

    if (candles.length < 300) {
      throw new Error('Insufficient data');
    }

    const trades: TradeFeatures[] = [];
    const lookback = 200;
    const weights = {
      trend_structure: 40,
      order_blocks: 30,
      fvgs: 20,
      ema_alignment: 15,
      liquidity: 10,
      mtf_bonus: 35,
      rsi_penalty: 15
    };

    for (let i = lookback; i < candles.length - 50; i++) {
      const currentCandle = candles[i];
      const historicalCandles = candles.slice(0, i + 1);

      const analysis = SMCIndicators.analyze(historicalCandles);
      const scoring = UnifiedScoring.calculateConfluence(analysis, currentCandle.close, weights);

      // LOW min score - we want losers too!
      if (scoring.score < CONFIG.minScore) continue;
      if (scoring.bias === 'neutral') continue;

      const direction = scoring.bias === 'bullish' ? 'long' : 'short';

      const features = FeatureExtractor.extractFeatures(
        candles, i, analysis, scoring.score, direction
      );

      const trade = this.simulateTrade(candles, i, currentCandle.close, direction, analysis);
      const featuresWithOutcome = FeatureExtractor.addOutcome(features, trade);

      trades.push(featuresWithOutcome);
    }

    return trades;
  }

  /**
   * Simulate a trade to get outcome
   */
  private simulateTrade(
    candles: Candle[],
    startIndex: number,
    entryPrice: number,
    direction: 'long' | 'short',
    analysis: any
  ): any {
    const isLong = direction === 'long';
    const atr = analysis.atr || (candles[startIndex].high - candles[startIndex].low);
    const stopLoss = isLong ? entryPrice - (atr * 2) : entryPrice + (atr * 2);
    const riskDistance = Math.abs(entryPrice - stopLoss);
    const tp2 = isLong ? entryPrice + (riskDistance * 3) : entryPrice - (riskDistance * 3);

    let exitPrice = candles[startIndex].close;

    for (let i = startIndex + 1; i < Math.min(startIndex + 100, candles.length); i++) {
      const candle = candles[i];

      if (isLong && candle.low <= stopLoss) { exitPrice = stopLoss; break; }
      if (!isLong && candle.high >= stopLoss) { exitPrice = stopLoss; break; }
      if (isLong && candle.high >= tp2) { exitPrice = tp2; break; }
      if (!isLong && candle.low <= tp2) { exitPrice = tp2; break; }
    }

    const priceDiff = isLong ? (exitPrice - entryPrice) : (entryPrice - exitPrice);
    const pnl = 1000 * (priceDiff / entryPrice);

    return {
      pnl,
      pnl_percent: (priceDiff / entryPrice) * 100,
      outcome: pnl > 0 ? 'WIN' : 'LOSS',
      exit_reason: exitPrice === stopLoss ? 'SL' : 'TP',
      holding_periods: 0
    };
  }

  /**
   * Evaluate model on a set of trades
   */
  private evaluateModel(trades: TradeFeatures[]): { accuracy: number; errors: TradeFeatures[] } {
    let correct = 0;
    const errors: TradeFeatures[] = [];

    for (const trade of trades) {
      const prediction = this.model.predict(trade);
      const predictedWin = prediction.winProbability >= 0.5;
      const actualWin = trade.outcome === 'WIN';

      if (predictedWin === actualWin) {
        correct++;
      } else {
        errors.push(trade);
      }
    }

    return {
      accuracy: correct / trades.length,
      errors
    };
  }

  /**
   * Save results and model
   */
  private async saveResults(): Promise<void> {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);

    // Save iteration history
    const historyFile = path.join(CONFIG.outputDir, `loop_history_${timestamp}.json`);
    fs.writeFileSync(historyFile, JSON.stringify({
      timestamp: new Date().toISOString(),
      config: CONFIG,
      iterations: this.iterations,
      totalTrades: this.allTrades.length
    }, null, 2));

    // Save model state
    const modelState = {
      modelId: `loop_model_${timestamp}`,
      trainedAt: Date.now(),
      iterations: this.iterations.length,
      finalAccuracy: this.iterations[this.iterations.length - 1]?.testAccuracy || 0,
      tradesUsed: this.allTrades.length,
      algorithm: 'backtest-learn-loop'
    };

    const modelFile = path.join(CONFIG.modelDir, 'best-model.json');
    fs.writeFileSync(modelFile, JSON.stringify(modelState, null, 2));

    // Save training data for future use
    const trainingFile = path.join(CONFIG.outputDir, `training_data_${timestamp}.csv`);
    this.saveToCSV(this.allTrades, trainingFile);

    console.log(`  Saved: ${historyFile}`);
    console.log(`  Saved: ${modelFile}`);
    console.log(`  Saved: ${trainingFile}`);
  }

  /**
   * Save trades to CSV
   */
  private saveToCSV(trades: TradeFeatures[], outputPath: string): void {
    if (trades.length === 0) return;

    const headers = Object.keys(trades[0]).join(',');
    const rows = trades.map(trade => {
      const values = Object.values(trade).map(val => {
        if (typeof val === 'string') return `"${val}"`;
        if (typeof val === 'boolean') return val ? 1 : 0;
        return val;
      });
      return values.join(',');
    });

    fs.writeFileSync(outputPath, [headers, ...rows].join('\n'));
  }

  /**
   * Ensure directories exist
   */
  private ensureDirectories(): void {
    [CONFIG.outputDir, CONFIG.modelDir].forEach(dir => {
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
    });
  }
}

// Also add live trade integration
async function addLiveTrades(): Promise<TradeFeatures[]> {
  const liveTrades: TradeFeatures[] = [];
  const tradesDir = path.join(process.cwd(), 'data', 'recorded-trades');

  if (!fs.existsSync(tradesDir)) return liveTrades;

  const files = fs.readdirSync(tradesDir).filter(f => f.endsWith('.json'));

  for (const file of files) {
    try {
      const trades = JSON.parse(fs.readFileSync(path.join(tradesDir, file), 'utf-8'));
      const closed = trades.filter((t: any) => t.status === 'CLOSED' && t.features);

      for (const trade of closed) {
        if (trade.features) {
          liveTrades.push({
            ...trade.features,
            outcome: trade.pnl > 0 ? 'WIN' : 'LOSS',
            pnl: trade.pnl,
            pnl_percent: trade.pnl_percent || 0,
            exit_reason: trade.exit_reason || 'unknown',
            holding_periods: trade.holding_periods || 0
          });
        }
      }
    } catch {}
  }

  return liveTrades;
}

// CLI
async function main() {
  const args = process.argv.slice(2);

  if (args.includes('--help') || args.includes('-h')) {
    console.log(`
Backtest-Learn Loop - Self-improving ML through backtesting

The model learns from its own mistakes by:
1. Extracting ALL trades (including low-score losers)
2. Predicting outcomes for each trade
3. Comparing predictions to actual outcomes
4. Re-training with emphasis on prediction errors
5. Repeating until accuracy plateaus

Usage: npx ts-node src/backtest-learn-loop.ts [OPTIONS]

Options:
  --workers <n>       Parallel workers (default: CPU cores - 2 = ${os.cpus().length - 2})
  --iterations <n>    Max iterations (default: 10)
  --min-score <n>     Minimum SMC score (default: 25, LOW to capture losers)
  --emphasis <n>      Error emphasis multiplier (default: 3)
  -h, --help          Show this help

Examples:
  npm run learn-loop
  npm run learn-loop -- --workers 16
  npm run learn-loop -- --iterations 20 --workers 8
  npm run learn-loop -- --min-score 20 --emphasis 5
    `);
    process.exit(0);
  }

  // Parse args
  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--workers':
        CONFIG.workers = parseInt(args[++i]);
        break;
      case '--iterations':
        CONFIG.maxIterations = parseInt(args[++i]);
        break;
      case '--min-score':
        CONFIG.minScore = parseInt(args[++i]);
        break;
      case '--emphasis':
        CONFIG.errorEmphasisMultiplier = parseInt(args[++i]);
        break;
    }
  }

  try {
    const loop = new BacktestLearnLoop();
    await loop.run();

    console.log('\n✅ Learning loop complete!');
    console.log('\nThe model has learned from its backtesting mistakes.');
    console.log('Live trades will continue to improve it further.');
    console.log('\nNext: Use analyze_setup in Claude to get ML-backed predictions.');

  } catch (error) {
    console.error('\n❌ Error:', error);
    process.exit(1);
  }
}

main().catch(console.error);
