export function cloneSimpleObject<T>(obj: T): T {
  return JSON.parse(JSON.stringify(obj));
}
