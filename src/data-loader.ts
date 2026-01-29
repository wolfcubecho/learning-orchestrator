/**
 * Local Data Loader
 * Loads historical candle data from CSV files
 */

import { Candle } from './smc-indicators.js';
import * as fs from 'fs';
import * as path from 'path';
import { exec } from 'child_process';

export interface DataLoadResult {
  symbol: string;
  interval: string;
  candles: Candle[];
  loaded: number;
  total: number;
}

export class LocalDataLoader {
  private dataPath: string;
  
  constructor(dataPath: string = './Historical_Data_Lite') {
    this.dataPath = dataPath;
  }
  
  /**
   * Find data file for symbol/timeframe - tries multiple patterns
   * Priority: lite folder -> parquet (archive) -> csv (filtered) -> csv (flat)
   */
  private findDataFile(symbol: string, interval: string): { path: string; type: 'parquet' | 'csv' } | null {
    // Get base path (remove /archive suffix if present for lite folder check)
    const basePath = this.dataPath.replace(/[\/\\]archive$/, '');

    const possiblePaths: { path: string; type: 'parquet' | 'csv' }[] = [
      // Historical_Data_Lite folder (highest priority - smaller, top 10 only)
      { path: path.join(basePath, '..', 'Historical_Data_Lite', interval, `${symbol}_${interval}.parquet`), type: 'parquet' },
      { path: path.join(this.dataPath, '..', 'Historical_Data_Lite', interval, `${symbol}_${interval}.parquet`), type: 'parquet' },

      // Parquet files from archive
      { path: path.join(this.dataPath, 'kilnes_TRADING', interval, 'TRADING', `${symbol}_${interval}.parquet`), type: 'parquet' },

      // Flat structure: Binance_BTCUSDT_1h.csv
      { path: path.join(this.dataPath, `Binance_${symbol}_${interval}.csv`), type: 'csv' },
      // Flat structure: BTCUSDT_1h.csv
      { path: path.join(this.dataPath, `${symbol}_${interval}.csv`), type: 'csv' },
      // Hierarchical: symbol/interval/BTCUSDT_1h.csv
      { path: path.join(this.dataPath, symbol, interval, `${symbol}_${interval}.csv`), type: 'csv' },
      // Hierarchical: symbol/interval/Binance_BTCUSDT_1h.csv
      { path: path.join(this.dataPath, symbol, interval, `Binance_${symbol}_${interval}.csv`), type: 'csv' },
      // With filtered/ prefix
      { path: path.join(this.dataPath, 'filtered', symbol, interval, `${symbol}_${interval}.csv`), type: 'csv' },
      { path: path.join(this.dataPath, 'filtered', symbol, interval, `Binance_${symbol}_${interval}.csv`), type: 'csv' },
    ];
    
    for (const filePath of possiblePaths) {
      if (fs.existsSync(filePath.path)) {
        return filePath;
      }
    }
    
    return null;
  }
  
  /**
   * Load candle data from CSV file
   * CSV format: timestamp,open,high,low,close,volume
   */
  private async loadCSV(fullPath: string): Promise<Candle[]> {
    
    return new Promise((resolve, reject) => {
      fs.readFile(fullPath, 'utf8', (err, data) => {
        if (err) {
          reject(err);
          return;
        }
        
        try {
          const lines = data.trim().split('\n');
          const candles: Candle[] = [];
          
          // Skip header row
          for (let i = 1; i < lines.length; i++) {
            const line = lines[i].trim();
            if (!line) continue;
            
            const parts = line.split(',');
            if (parts.length < 5) continue;
            
            const candle: Candle = {
              timestamp: parseInt(parts[0]),
              open: parseFloat(parts[1]),
              high: parseFloat(parts[2]),
              low: parseFloat(parts[3]),
              close: parseFloat(parts[4]),
              volume: parseFloat(parts[5])
            };
            
            // Validate data
            if (candle.open > 0 && candle.high > 0 && candle.low > 0 && candle.close > 0) {
              candles.push(candle);
            }
          }
          
          resolve(candles);
        } catch (error) {
          reject(error);
        }
      });
    });
  }
  
  /**
   * Load candle data from Parquet file using Python script
   */
  private async loadParquet(fullPath: string): Promise<Candle[]> {
    return new Promise((resolve, reject) => {
      // Get script path - resolve relative to current working directory
      const scriptPath = path.join(process.cwd(), 'scripts', 'read-parquet.py');
      
      exec(`python "${scriptPath}" "${fullPath}"`, { maxBuffer: 100 * 1024 * 1024 }, (error, stdout, stderr) => {
        if (error) {
          reject(new Error(`Failed to read parquet: ${error.message}\n${stderr}`));
          return;
        }
        
        try {
          const result = JSON.parse(stdout);
          if (result.success) {
            resolve(result.candles);
          } else {
            reject(new Error(result.error || 'Failed to parse parquet'));
          }
        } catch (e) {
          reject(new Error(`Failed to parse Python output: ${e}`));
        }
      });
    });
  }
  
  /**
   * Load data for a specific symbol and interval
   */
  async loadData(symbol: string, interval: string): Promise<DataLoadResult> {
    // Find the data file
    const fileData = this.findDataFile(symbol, interval);
    
    if (!fileData) {
      throw new Error(
        `Data file not found for ${symbol} ${interval}\n` +
        `Looked in: ${this.dataPath}\n` +
        `Expected patterns: ${symbol}_${interval}.parquet or ${symbol}_${interval}.csv\n` +
        `Archive path: Historical_Data/archive/kilnes_TRADING/${interval}/TRADING/`
      );
    }
    
    console.log(`Found data file: ${fileData.path} (${fileData.type})`);
    
    let candles: Candle[];
    if (fileData.type === 'parquet') {
      // Convert to absolute path
      const absolutePath = path.resolve(fileData.path);
      candles = await this.loadParquet(absolutePath);
    } else {
      candles = await this.loadCSV(fileData.path);
    }
    
    return {
      symbol,
      interval,
      candles,
      loaded: candles.length,
      total: candles.length
    };
  }
  
  /**
   * Load data for multiple symbols
   */
  async loadMulti(symbols: string[], interval: string): Promise<Map<string, Candle[]>> {
    const results = new Map<string, Candle[]>();
    
    const promises = symbols.map(symbol =>
      this.loadData(symbol, interval)
        .then(result => results.set(symbol, result.candles))
        .catch(err => {
          console.warn(`Failed to load ${symbol}: ${err.message}`);
          return null;
        })
    );
    
    await Promise.all(promises);
    
    return results;
  }
  
  /**
   * Get available symbols and intervals
   */
  getAvailableData(): { symbols: string[]; intervals: string[] } {
    const symbols = new Set<string>();
    const intervals = new Set<string>();
    
    // Check directory structure
    const dataDir = this.dataPath;
    
    if (fs.existsSync(dataDir)) {
      const files = fs.readdirSync(dataDir);
      
      files.forEach(file => {
        if (file.endsWith('.csv')) {
          // Parse filename to extract symbol and interval
          const match = file.match(/(\w+)[_-](\d+[dhm])\.csv/);
          if (match) {
            symbols.add(match[1].replace('USDT', 'USDT')); // Normalize
            intervals.add(match[2]);
          }
        }
      });
    }
    
    return {
      symbols: Array.from(symbols),
      intervals: Array.from(intervals)
    };
  }
}