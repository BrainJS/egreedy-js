import { MersenneTwister19937, Random } from 'random-js';

const random = new Random(MersenneTwister19937.autoSeed());

export interface IEGreedyOptions {
  arms?: number | string;
  epsilon?: number | string;
  counts?: number[];
  values?: number[];
}

export interface IEGreedySerialized {
  arms: number;
  epsilon: number;
  counts: number[];
  values: number[];
}

export class EGreedy implements IEGreedySerialized {
  arms: number;
  epsilon: number;
  counts: number[];
  values: number[];

  constructor(options: IEGreedyOptions = {}) {
    this.arms = options.arms === undefined ? 2 : parseInt(options.arms as string, 10);
    this.epsilon = options.epsilon === undefined ? 0.5 : parseFloat(options.epsilon as string);

    if (this.arms < 1) {
      throw new TypeError('invalid arms: cannot be less than 1');
    } else if (this.epsilon < 0) {
      throw new TypeError('invalid epsilon: cannot be less than 0');
    } else if (this.epsilon > 1) {
      throw new TypeError('invalid epsilon: cannot be greater than 1');
    }

    const serialized = options as IEGreedySerialized;
    if (typeof serialized.counts !== 'undefined' && typeof serialized.values !== 'undefined') {
      if (!Array.isArray(options.counts)) {
        throw new TypeError('counts must be an array');
      } else if (!Array.isArray(options.values)) {
        throw new TypeError('values must be an array');
      } else if (options.counts.length !== this.arms) {
        throw new Error('arms and counts.length must be identical');
      } else if (options.values.length !== this.arms) {
        throw new Error('arms and values.length must be identical');
      }

      this.counts = options.counts.slice(0);
      this.values = options.values.slice(0);
    } else {
      this.counts = new Array(this.arms).fill(0);
      this.values = new Array(this.arms).fill(0);
    }
  }

  async reward(arm: number, value: number): Promise<this> {
    return this.rewardSync(arm, value);
  }

  rewardSync(arm: number, value: number): this {
    if (typeof arm !== 'number') {
      throw new TypeError('missing or invalid required parameter: arm');
    } else if (arm >= this.arms || arm < 0) {
      throw new TypeError('arm index out of bounds');
    } else if (typeof value !== 'number') {
      throw new TypeError('missing or invalid required parameter: reward');
    }

    const count = this.counts[arm] + 1;
    const prior = this.values[arm];

    this.counts[arm] = count;
    this.values[arm] = (((count - 1) / count) * prior) + ((1 / count) * value);

    return this;
  }

  async select(): Promise<number> {
    return this.selectSync();
  }

  selectSync(): number {
    const r = random.real(0, 1, true);
    const n = sum(this.counts);

    if (this.epsilon > r || n === 0) {
      return random.integer(0, this.arms - 1);
    }

    return this.values.indexOf(Math.max.apply(null, this.values));
  }

  async serialize(): Promise<IEGreedySerialized> {
    return this.serializeSync();
  }

  serializeSync(): IEGreedySerialized {
    return {
      arms: this.arms,
      epsilon: this.epsilon,
      counts: this.counts.slice(0),
      values: this.values.slice(0)
    };
  }
}

function sum(arr: number[]): number {
  return arr.reduce(add);
}

function add(out: number, value: number): number {
  return out + value;
}
