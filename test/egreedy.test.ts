import { MersenneTwister19937, Random } from 'random-js';
import { EGreedy, IEGreedyOptions } from '../src/egreedy';
import { cloneSimpleObject } from './utils/clone-simple-object';

const random = new Random(MersenneTwister19937.autoSeed());

describe('EGreedy', () => {
  describe('constructor', () => {
    const arms = random.integer(2, 10);
    const epsilon = random.real(0, 1);
    const state: IEGreedyOptions = {
      arms,
      counts: new Array(arms).fill(0),
      values: new Array(arms).fill(0)
    };

    it('does not require new keyword', () => {
      const alg = new EGreedy();

      expect(alg.arms).toBeDefined();
      expect(alg.counts).toBeDefined();
      expect(alg.values).toBeDefined();
    });

    it('restores instance properties', () => {
      const alg = new EGreedy({ ...state  });

      expect(alg.arms).toEqual(state.arms);
      expect(alg.epsilon).toEqual(0.5);
      expect(alg.counts).toEqual(state.counts);
      expect(alg.values).toEqual(state.values);
    });

    it('restores instance properties (with epsilon)', () => {
      const alg = new EGreedy({ epsilon, ...state });

      expect(alg.arms).toEqual(state.arms);
      expect(alg.epsilon).toEqual(epsilon);
      expect(alg.counts).toEqual(state.counts);
      expect(alg.values).toEqual(state.values);
    });

    it('throws TypeError when passed arms=0', () => {
      function test() {
        return new EGreedy({ arms: 0 });
      }

      expect(test).toThrow(TypeError);
      expect(test).toThrow(/invalid arms: cannot be less than 1/);
    });

    it('throws TypeError when passed arms<0', () => {
      function test() {
        return new EGreedy({ arms: -1 });
      }

      expect(test).toThrow(TypeError);
      expect(test).toThrow(/invalid arms: cannot be less than 1/);
    });

    it('throws TypeError when passed epsilon<0', () => {
      function test() {
        return new EGreedy({ epsilon: -1 });
      }

      expect(test).toThrow(TypeError);
      expect(test).toThrow(/invalid epsilon: cannot be less than 0/);
    });

    it('throws TypeError when passed epsilon<0', () => {
      function test() {
        return new EGreedy({ epsilon: 2 });
      }

      expect(test).toThrow(TypeError);
      expect(test).toThrow(/invalid epsilon: cannot be greater than 1/);
    });

    it('throws if counts is not an array', () => {
      const localState: IEGreedyOptions = {
        ...state,
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-ignore
        counts: Date.now().toString(16),
      };

      function test() {
        return new EGreedy(localState);
      }

      expect(test).toThrow(TypeError);
      expect(test).toThrow(/counts must be an array/);
    });

    it('throws if values is not an array', () => {
      const localState = {
        ...state,
        values: Date.now().toString(16),
      };

      function test() {
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-ignore
        return new EGreedy(localState);
      }

      expect(test).toThrow(TypeError);
      expect(test).toThrow(/values must be an array/);
    });

    it('throws if counts.length does not equal arm count', () => {
      const localState = cloneSimpleObject(state);

      localState.counts.pop();

      function test() {
        return new EGreedy(localState);
      }

      expect(test).toThrow(Error);
      expect(test).toThrow(/arms and counts.length must be identical/);
    });

    it('throws if values.length does not equal arm count', () => {
      const localState = cloneSimpleObject(state);

      localState.values.pop();

      function test() {
        return new EGreedy(localState);
      }

      expect(test).toThrow(Error);
      expect(test).toThrow(/arms and values.length must be identical/);
    });
  });

  describe('reward', () => {
    const arms = random.integer(2, 10);
    const config = {
      arms,
      epsilon: random.real(0, 1)
    };

    it('updates the values and counts accumulators', () => {
      const alg = new EGreedy(config);

      const arm = random.integer(0, arms - 1);
      const val = random.integer(0, 100) / 100;

      return alg.reward(arm, val).then(() => {
        expect(alg.counts[arm]).toEqual(1);
        expect(alg.values[arm]).toEqual(val);

        expect(alg.counts.reduce((accum, x) => accum + x)).toEqual(1);
        expect(alg.values.reduce((accum, x) => accum + x)).toEqual(val);
      });
    });

    it('updates the observation counter', () => {
      const alg = new EGreedy(config);

      const arm = random.integer(0, arms - 1);
      const val = random.integer(0, 100) / 100;

      const pre = alg.counts.reduce((out, x) => out + x);

      return alg.reward(arm, val).then(() => {
        const post = alg.counts.reduce((accum, x) => accum + x);

        expect(post).toEqual(pre + 1);
      });
    });

    it('resolves to the updated algorithm instance', () => {
      const alg = new EGreedy(config);

      const arm = random.integer(0, arms - 1);
      const val = random.integer(0, 100) / 100;

      return alg.reward(arm, val).then((out) => {
        expect(out).toBeInstanceOf(EGreedy);

        expect(out.select).toBeInstanceOf(Function);
        expect(out.reward).toBeInstanceOf(Function);
        expect(out.serialize).toBeInstanceOf(Function);
      });
    });

    it('throws if the arm index is null', async () => {
      const alg = new EGreedy(config);

      const val = random.integer(0, 100) / 100;

      await expect(alg.reward(null as number, val)).rejects.toEqual(
       new TypeError('missing or invalid required parameter: arm')
      );
    });

    it('throws if the arm index is negative', async () => {
      const alg = new EGreedy(config);

      const val = random.integer(0, 100) / 100;

      await expect(alg.reward(-1, val)).rejects.toEqual(
        new TypeError('arm index out of bounds')
      );
    });

    it('throws if the arm index exceeds total arms', async () => {
      const alg = new EGreedy(config);

      const val = random.integer(0, 100) / 100;

      await expect(alg.reward(config.arms * 10, val)).rejects.toEqual(
        new TypeError('arm index out of bounds')
      );
    });

    it('throws if the arm index is undefined', async () => {
      const alg = new EGreedy(config);

      const val = random.integer(0, 100) / 100;

      // eslint-disable-next-line @typescript-eslint/ban-ts-comment
      // @ts-ignore
      await expect(alg.reward(undefined, val)).rejects.toEqual(
        new TypeError('missing or invalid required parameter: arm')
      );
    });

    it('throws if the arm index is not a number', async () => {
      const alg = new EGreedy(config);

      const val = random.integer(0, 100) / 100;

      // eslint-disable-next-line @typescript-eslint/ban-ts-comment
      // @ts-ignore
      await expect(alg.reward('0', val)).rejects.toEqual(
        new TypeError('missing or invalid required parameter: arm')
      );
    });

    it('throws if the reward is null', async () => {
      const alg = new EGreedy(config);

      await expect(alg.reward(0, null)).rejects.toEqual(
        new TypeError('missing or invalid required parameter: reward')
      );
    });

    it('throws if the reward is undefined', async () => {
      const alg = new EGreedy(config);

      await expect(alg.reward(0, undefined)).rejects.toEqual(
        new TypeError('missing or invalid required parameter: reward')
      );
    });

    it('throws if the reward is not a number', async () => {
      const alg = new EGreedy(config);

      // eslint-disable-next-line @typescript-eslint/ban-ts-comment
      // @ts-ignore
      await expect(alg.reward(0, '1')).rejects.toEqual(
        new TypeError('missing or invalid required parameter: reward')
      );
    });
  });

  describe('select', () => {
    const arms = random.integer(2, 10);
    const config = {
      arms,
      epsilon: random.real(0.6, 0.8)
    };

    it('returns a number', async () => {
      const alg = new EGreedy(config);

      const arm = await alg.select();
      expect(typeof arm).toBe('number');
    });

    it('returns a valid arm', async () => {
      const alg = new EGreedy(config);

      const trials = new Array(random.integer(10, 20)).fill(-1);

      const selections = await Promise.all(trials.map(() => alg.select()));
      expect(selections.length).toEqual(trials.length);
      selections.forEach((choice) => {
        expect(typeof choice).toBe('number');
        expect(choice).toBeLessThan(arms);
      });
    });

    it('begins to exploit best arm (first)', async () => {
      const alg = new EGreedy(config);

      for (let i = 0; i < arms * 100; i++) {
        const arm = await alg.select();
        await alg.reward(arm, arm === 0 ? 1 : 0);
      }

      const bestCt = alg.counts[0];
      alg.counts.slice(1).forEach((ct) => {
        expect(ct).toBeLessThan(bestCt);
      });
    });

    it('begins to exploit best arm (last)', async () => {
      const alg = new EGreedy(config);

      for (let i = 0; i < arms * 100; i++) {
        const arm = await alg.select();
        await alg.reward(arm, arm === (arms - 1) ? 1 : 0);
      }

      const bestCt = alg.counts[arms - 1];

      alg.counts.slice(0, -1).forEach((ct) => {
        expect(ct).toBeLessThan(bestCt);
      });
    });
  });

  describe('serialize', () => {
    const arms = random.integer(2, 20);
    const config = {
      arms
    };

    const emptyArray = new Array(arms).fill(0);

    it('returns a valid state', async () => {
      const alg = new EGreedy(config);

      const state = await alg.serialize();
      expect(state.arms).toEqual(arms);
      expect(state.counts).toEqual(emptyArray);
      expect(state.values).toEqual(emptyArray);
    });
  });
});
