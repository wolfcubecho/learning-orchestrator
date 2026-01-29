/**
 * SMC (Smart Money Concepts) Technical Indicators
 * Calculates Order Blocks, Fair Value Gaps, EMAs, Liquidity, etc.
 */

export interface Candle {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface OrderBlock {
  index: number;
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  type: 'bull' | 'bear';
  strength: number;
}

export interface FVG {
  index: number;
  from: number;
  to: number;
  type: 'bull' | 'bear';
  size: number;
}

export interface LiquidityZones {
  highs: Array<{ price: number; touches: number; lastTouch: number }>;
  lows: Array<{ price: number; touches: number; lastTouch: number }>;
}

export interface SMCAnalysis {
  trend: 'up' | 'down' | null;
  bos: 'up' | 'down' | null;
  ema50: number | null;
  ema200: number | null;
  rsi: number | null;
  orderBlocks: OrderBlock[];
  fvg: FVG[];
  liquidityZones: LiquidityZones;
  atr: number | null;
  vwap: number | null;
}

export class SMCIndicators {
  /**
   * Calculate Simple Moving Average
   */
  static sma(data: number[], period: number): number[] {
    const result: number[] = [];
    for (let i = period - 1; i < data.length; i++) {
      const sum = data.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
      result.push(sum / period);
    }
    return result;
  }

  /**
   * Calculate Exponential Moving Average
   */
  static ema(data: number[], period: number): number[] {
    const result: number[] = [];
    const multiplier = 2 / (period + 1);
    
    // Start with SMA for first value
    let ema = data.slice(0, period).reduce((a, b) => a + b, 0) / period;
    result.push(ema);
    
    for (let i = period; i < data.length; i++) {
      ema = (data[i] - ema) * multiplier + ema;
      result.push(ema);
    }
    
    return result;
  }

  /**
   * Calculate RSI
   */
  static rsi(candles: Candle[], period: number = 14): number[] {
    const result: number[] = [];
    const closes = candles.map(c => c.close);
    
    for (let i = period; i < closes.length; i++) {
      const slice = closes.slice(i - period, i);
      const gains: number[] = [];
      const losses: number[] = [];
      
      for (let j = 1; j < slice.length; j++) {
        const diff = slice[j] - slice[j - 1];
        if (diff >= 0) {
          gains.push(diff);
          losses.push(0);
        } else {
          gains.push(0);
          losses.push(Math.abs(diff));
        }
      }
      
      const avgGain = gains.reduce((a, b) => a + b, 0) / period;
      const avgLoss = losses.reduce((a, b) => a + b, 0) / period;
      
      if (avgLoss === 0) {
        result.push(100);
      } else {
        const rs = avgGain / avgLoss;
        result.push(100 - (100 / (1 + rs)));
      }
    }
    
    return result;
  }

  /**
   * Calculate ATR (Average True Range)
   */
  static atr(candles: Candle[], period: number = 14): number[] {
    const result: number[] = [];
    
    for (let i = period - 1; i < candles.length; i++) {
      const trueRanges: number[] = [];
      
      for (let j = i - period + 1; j <= i; j++) {
        const c = candles[j];
        const tr = Math.max(
          c.high - c.low,
          Math.abs(c.high - c.close),
          Math.abs(c.low - c.close)
        );
        trueRanges.push(tr);
      }
      
      result.push(trueRanges.reduce((a, b) => a + b, 0) / period);
    }
    
    return result;
  }

  /**
   * Calculate VWAP
   */
  static vwap(candles: Candle[]): number[] {
    const result: number[] = [];
    let cumVolPrice = 0;
    let cumVol = 0;
    
    for (let i = 0; i < candles.length; i++) {
      const c = candles[i];
      const typicalPrice = (c.high + c.low + c.close) / 3;
      cumVolPrice += typicalPrice * c.volume;
      cumVol += c.volume;
      result.push(cumVolPrice / cumVol);
    }
    
    return result;
  }

  /**
   * Detect Order Blocks (last strong candle before a move)
   */
  static detectOrderBlocks(candles: Candle[], lookback: number = 10): OrderBlock[] {
    const orderBlocks: OrderBlock[] = [];
    
    for (let i = lookback; i < candles.length - 2; i++) {
      const candle = candles[i];
      const nextCandle = candles[i + 1];
      const nextNextCandle = candles[i + 2];
      
      const bodySize = Math.abs(candle.close - candle.open);
      const avgBody = candles.slice(i - lookback, i)
        .reduce((sum, c) => sum + Math.abs(c.close - c.open), 0) / lookback;
      
      // Bullish OB: strong bearish candle followed by strong bullish move
      if (candle.close < candle.open && bodySize > avgBody * 1.5) {
        if (nextCandle.close > nextCandle.open && 
            nextNextCandle.close > nextNextCandle.open) {
          orderBlocks.push({
            index: i,
            timestamp: candle.timestamp,
            open: candle.open,
            high: candle.high,
            low: candle.low,
            close: candle.close,
            type: 'bull',
            strength: bodySize / avgBody
          });
        }
      }
      
      // Bearish OB: strong bullish candle followed by strong bearish move
      if (candle.close > candle.open && bodySize > avgBody * 1.5) {
        if (nextCandle.close < nextCandle.open && 
            nextNextCandle.close < nextNextCandle.open) {
          orderBlocks.push({
            index: i,
            timestamp: candle.timestamp,
            open: candle.open,
            high: candle.high,
            low: candle.low,
            close: candle.close,
            type: 'bear',
            strength: bodySize / avgBody
          });
        }
      }
    }
    
    return orderBlocks;
  }

  /**
   * Detect Fair Value Gaps
   */
  static detectFVG(candles: Candle[]): FVG[] {
    const fvgList: FVG[] = [];
    
    for (let i = 1; i < candles.length - 1; i++) {
      const prev = candles[i - 1];
      const curr = candles[i];
      const next = candles[i + 1];
      
      // Bullish FVG: gap between prev candle's high and next candle's low
      if (prev.high < next.low) {
        const size = next.low - prev.high;
        fvgList.push({
          index: i,
          from: prev.high,
          to: next.low,
          type: 'bull',
          size
        });
      }
      
      // Bearish FVG: gap between prev candle's low and next candle's high
      if (prev.low > next.high) {
        const size = prev.low - next.high;
        fvgList.push({
          index: i,
          from: next.high,
          to: prev.low,
          type: 'bear',
          size
        });
      }
    }
    
    return fvgList;
  }

  /**
   * Detect Liquidity Zones (swing highs and lows)
   */
  static detectLiquidity(candles: Candle[], swingPeriod: number = 5): LiquidityZones {
    const highs: Array<{ price: number; touches: number; lastTouch: number }> = [];
    const lows: Array<{ price: number; touches: number; lastTouch: number }> = [];
    
    // Find swing highs
    for (let i = swingPeriod; i < candles.length - swingPeriod; i++) {
      const isSwingHigh = candles.slice(i - swingPeriod, i + swingPeriod + 1)
        .every(c => c.high <= candles[i].high);
      
      if (isSwingHigh) {
        highs.push({
          price: candles[i].high,
          touches: 0,
          lastTouch: i
        });
      }
      
      const isSwingLow = candles.slice(i - swingPeriod, i + swingPeriod + 1)
        .every(c => c.low >= candles[i].low);
      
      if (isSwingLow) {
        lows.push({
          price: candles[i].low,
          touches: 0,
          lastTouch: i
        });
      }
    }
    
    return { highs, lows };
  }

  /**
   * Determine trend using SMA crossovers
   */
  static getTrend(candles: Candle[]): 'up' | 'down' | null {
    if (candles.length < 200) return null;
    
    const closes = candles.map(c => c.close);
    const sma50 = this.sma(closes, 50);
    const sma200 = this.sma(closes, 200);
    
    const lastSMA50 = sma50[sma50.length - 1];
    const lastSMA200 = sma200[sma200.length - 1];
    
    if (lastSMA50 > lastSMA200) return 'up';
    if (lastSMA50 < lastSMA200) return 'down';
    return null;
  }

  /**
   * Detect Break of Structure (BOS)
   */
  static getBOS(candles: Candle[]): 'up' | 'down' | null {
    if (candles.length < 50) return null;
    
    const closes = candles.map(c => c.close);
    const sma50 = this.sma(closes, 50);
    const prevSMA50 = sma50[sma50.length - 2];
    const lastSMA50 = sma50[sma50.length - 1];
    
    if (lastSMA50 > prevSMA50) return 'up';
    if (lastSMA50 < prevSMA50) return 'down';
    return null;
  }

  /**
   * Perform full SMC analysis on candles
   */
  static analyze(candles: Candle[]): SMCAnalysis {
    const closes = candles.map(c => c.close);
    
    const ema50Arr = this.ema(closes, 50);
    const ema200Arr = this.ema(closes, 200);
    const rsiArr = this.rsi(candles, 14);
    const atrArr = this.atr(candles, 14);
    const vwapArr = this.vwap(candles);
    
    return {
      trend: this.getTrend(candles),
      bos: this.getBOS(candles),
      ema50: ema50Arr[ema50Arr.length - 1] || null,
      ema200: ema200Arr[ema200Arr.length - 1] || null,
      rsi: rsiArr[rsiArr.length - 1] || null,
      orderBlocks: this.detectOrderBlocks(candles),
      fvg: this.detectFVG(candles),
      liquidityZones: this.detectLiquidity(candles),
      atr: atrArr[atrArr.length - 1] || null,
      vwap: vwapArr[vwapArr.length - 1] || null
    };
  }

  /**
   * Score a setup based on SMC factors
   */
  static scoreSetup(
    analysis: SMCAnalysis,
    currentPrice: number,
    weights: Record<string, number>
  ): { score: number; breakdown: Record<string, number>; confluence: string[] } {
    const breakdown: Record<string, number> = {};
    const confluence: string[] = [];
    let totalScore = 0;

    // Trend Structure (0-40 points)
    if (analysis.trend && analysis.bos && analysis.trend === analysis.bos) {
      breakdown.trend_structure = weights.trend_structure;
      confluence.push(`${analysis.trend} trend + BOS aligned`);
    } else if (analysis.trend) {
      breakdown.trend_structure = weights.trend_structure * 0.5;
      confluence.push(`${analysis.trend} trend`);
    } else {
      breakdown.trend_structure = 0;
    }
    totalScore += breakdown.trend_structure || 0;

    // Order Blocks (0-30 points)
    const relevantOBs = analysis.orderBlocks.filter(ob => {
      const obHigh = Math.max(ob.open, ob.close);
      const obLow = Math.min(ob.open, ob.close);
      return currentPrice >= obLow * 0.95 && currentPrice <= obHigh * 1.05;
    });

    if (relevantOBs.length > 0) {
      const bullOBs = relevantOBs.filter(ob => ob.type === 'bull');
      const bearOBs = relevantOBs.filter(ob => ob.type === 'bear');
      
      if (analysis.trend === 'up' && bullOBs.length > 0) {
        breakdown.order_blocks = weights.order_blocks;
        confluence.push(`${bullOBs.length} bullish order block(s)`);
      } else if (analysis.trend === 'down' && bearOBs.length > 0) {
        breakdown.order_blocks = weights.order_blocks;
        confluence.push(`${bearOBs.length} bearish order block(s)`);
      } else {
        breakdown.order_blocks = weights.order_blocks * 0.3;
        confluence.push(`${relevantOBs.length} order block(s) (mixed/aligned)`);
      }
    } else {
      breakdown.order_blocks = 0;
    }
    totalScore += breakdown.order_blocks || 0;

    // FVGs (0-20 points)
    const relevantFVGs = analysis.fvg.filter(fvg => {
      return currentPrice >= fvg.from * 0.98 && currentPrice <= fvg.to * 1.02;
    });

    if (relevantFVGs.length > 0) {
      breakdown.fvgs = weights.fvgs;
      confluence.push(`${relevantFVGs.length} FVG(s)`);
    } else {
      breakdown.fvgs = 0;
    }
    totalScore += breakdown.fvgs || 0;

    // EMA Alignment (0-15 points)
    if (analysis.ema50 && analysis.ema200) {
      const emaTrend = analysis.ema50 > analysis.ema200 ? 'up' : 'down';
      if (analysis.trend === emaTrend) {
        breakdown.ema_alignment = weights.ema_alignment;
        confluence.push(`EMA aligned ${emaTrend}`);
      } else {
        breakdown.ema_alignment = 0;
      }
    } else {
      breakdown.ema_alignment = 0;
    }
    totalScore += breakdown.ema_alignment || 0;

    // Liquidity (0-10 points)
    const hasLiquidity = analysis.liquidityZones.highs.length > 0 || 
                         analysis.liquidityZones.lows.length > 0;
    if (hasLiquidity) {
      breakdown.liquidity = weights.liquidity;
      confluence.push(`Liquidity zones present`);
    } else {
      breakdown.liquidity = 0;
    }
    totalScore += breakdown.liquidity || 0;

    // MTF Bonus (0-35 points)
    breakdown.mtf_bonus = weights.mtf_bonus;
    totalScore += breakdown.mtf_bonus;

    // RSI Penalty (-15 to 0)
    if (analysis.rsi) {
      if (analysis.rsi > 70) {
        breakdown.rsi_penalty = weights.rsi_penalty;
        confluence.push(`RSI overbought (${analysis.rsi.toFixed(1)})`);
      } else if (analysis.rsi < 30) {
        breakdown.rsi_penalty = weights.rsi_penalty * 0.5;
        confluence.push(`RSI oversold (${analysis.rsi.toFixed(1)})`);
      } else {
        breakdown.rsi_penalty = 0;
      }
    } else {
      breakdown.rsi_penalty = 0;
    }
    totalScore += breakdown.rsi_penalty || 0;

    return { score: Math.max(0, totalScore), breakdown, confluence };
  }
}