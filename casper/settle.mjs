#!/usr/bin/env node
// Casper Testnet native CSPR transfer settler.
//
// Builds, signs, and (unless --dry-run) broadcasts a native transfer on a
// Casper 2.x ("Condor") network using casper-js-sdk v5's NativeTransferBuilder.
// Prints exactly ONE line of JSON to stdout:
//   {"ok":true,"hash":"...","explorer":"https://testnet.cspr.live/deploy/<hash>", ...}
//   {"ok":false,"error":"..."}
//
// Usage:
//   node settle.mjs --pem <ed25519 pkcs8 pem> --to <casper pubkey hex 01..> \
//        --motes <amount> [--chain casper-test] [--rpc <url>] [--memo <id>] [--dry-run]
//
// Note: Casper native transfers have a protocol-enforced minimum of
// 2_500_000_000 motes (2.5 CSPR).

import fs from "node:fs";

const MIN_TRANSFER_MOTES = 2_500_000_000n;
// Standard cost of a native transfer on Casper 2.x: 0.1 CSPR.
const TRANSFER_PAYMENT_MOTES = 100_000_000;
const DEFAULT_RPC = "https://node.testnet.casper.network/rpc";
const DEFAULT_CHAIN = "casper-test";

function parseArgs(argv) {
  const out = { chain: DEFAULT_CHAIN, rpc: DEFAULT_RPC, dryRun: false, memo: undefined };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    switch (a) {
      case "--pem": out.pem = argv[++i]; break;
      case "--to": out.to = argv[++i]; break;
      case "--motes": out.motes = argv[++i]; break;
      case "--chain": out.chain = argv[++i]; break;
      case "--rpc": out.rpc = argv[++i]; break;
      case "--memo": out.memo = argv[++i]; break;
      case "--dry-run": out.dryRun = true; break;
      default: throw new Error(`unknown argument: ${a}`);
    }
  }
  if (!out.pem) throw new Error("--pem <path> is required");
  if (!out.to) throw new Error("--to <recipient public key hex> is required");
  if (!out.motes) throw new Error("--motes <amount> is required");
  return out;
}

function emit(obj) {
  process.stdout.write(JSON.stringify(obj) + "\n");
}

async function main() {
  const args = parseArgs(process.argv);

  const motes = BigInt(args.motes);
  if (motes < MIN_TRANSFER_MOTES) {
    throw new Error(
      `Casper native transfers require at least ${MIN_TRANSFER_MOTES} motes (2.5 CSPR); got ${motes}`
    );
  }

  // casper-js-sdk v5 ships CJS; the ESM namespace puts everything on `default`.
  const sdk = (await import("casper-js-sdk")).default;
  const { PrivateKey, PublicKey, KeyAlgorithm, NativeTransferBuilder, RpcClient, HttpHandler } = sdk;

  const pem = fs.readFileSync(args.pem, "utf8");
  const sender = PrivateKey.fromPem(pem, KeyAlgorithm.ED25519);
  const recipient = PublicKey.fromHex(args.to);

  // Transfer id: numeric memo, defaults to current time in ms.
  const transferId = args.memo !== undefined ? Number(args.memo) : Date.now();

  const transaction = new NativeTransferBuilder()
    .from(sender.publicKey)
    .target(recipient)
    .amount(motes.toString())
    .id(transferId)
    .chainName(args.chain)
    .payment(TRANSFER_PAYMENT_MOTES)
    .build();

  transaction.sign(sender);
  transaction.validate();

  const localHash = transaction.hash.toHex();

  if (args.dryRun) {
    emit({
      ok: true,
      dryRun: true,
      hash: localHash,
      explorer: `https://testnet.cspr.live/deploy/${localHash}`,
      from: sender.publicKey.toHex(),
      to: args.to,
      motes: motes.toString(),
      chain: args.chain,
    });
    return;
  }

  const rpcClient = new RpcClient(new HttpHandler(args.rpc));
  const result = await rpcClient.putTransaction(transaction);
  const hash = result.transactionHash.toHex?.() ?? localHash;

  emit({
    ok: true,
    hash,
    explorer: `https://testnet.cspr.live/deploy/${hash}`,
    from: sender.publicKey.toHex(),
    to: args.to,
    motes: motes.toString(),
    chain: args.chain,
  });
}

main().catch((err) => {
  emit({ ok: false, error: String(err?.message ?? err) });
  process.exit(1);
});
