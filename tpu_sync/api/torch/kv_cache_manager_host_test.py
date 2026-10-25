# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host-only tests for the Python pool-admission surface."""

import unittest

from tpu_sync.api.torch import kv_cache_manager
from tpu_sync.api.torch import pool_layout


class KVCacheManagerHostTest(unittest.TestCase):

  def test_aliased_pool_admits_last_live_region_without_trailing_padding(self):
    manager = kv_cache_manager.KVCacheManager.create_host_only_for_testing(
        num_layers=1,
        num_shards=1,
        slice_byte_size=128,
        node_id=7,
        host_blocks=2,
        parallelism=1,
    )
    pool = pool_layout.PoolSpec(
        tag="aliased",
        storage_index=0,
        base_offset_bytes=32,
        block_stride_bytes=128,
        num_blocks=2,
        regions=(
            pool_layout.RegionSpec(
                name="payload",
                offset_bytes=0,
                stride_bytes=64,
                unit_bytes=32,
                num_units=2,
            ),
        ),
    )

    self.assertEqual(pool.storage_extent_end_bytes, 256)
    pool.validate(storage_bytes=256)
    self.assertTrue(manager.register_pools((pool,))["admitted"])
    first = manager.get_block_ref(pool_idx=0, block_id=0)
    last = manager.get_block_ref(pool_idx=0, block_id=1)
    self.assertEqual(last["ptr"], first["ptr"] + 128)

  def test_register_refs_tags_and_admission_summary(self):
    manager = kv_cache_manager.KVCacheManager.create_host_only_for_testing(
        num_layers=1,
        num_shards=1,
        slice_byte_size=256,
        node_id=7,
        host_blocks=4,
        parallelism=1,
    )
    first_pool = pool_layout.PoolSpec(
        tag="kind_a",
        storage_index=0,
        base_offset_bytes=32,
        block_stride_bytes=128,
        num_blocks=4,
        regions=(
            pool_layout.RegionSpec(
                name="payload",
                offset_bytes=0,
                stride_bytes=128,
                unit_bytes=128,
                num_units=1,
            ),
        ),
        dtype_tag="dtype_a",
    )
    second_pool = pool_layout.PoolSpec(
        tag="kind_b",
        storage_index=0,
        base_offset_bytes=640,
        block_stride_bytes=64,
        num_blocks=4,
        regions=(
            pool_layout.RegionSpec(
                name="payload",
                offset_bytes=0,
                stride_bytes=64,
                unit_bytes=64,
                num_units=1,
            ),
        ),
        dtype_tag="dtype_b",
    )

    summary = manager.register_pools((first_pool, second_pool))

    self.assertEqual(
        summary,
        {
            "admitted": True,
            "pools": 2,
            "storages": 1,
            "tags": {"kind_a": 1, "kind_b": 1},
            "host_staging": {
                "mode": "full",
                "leases": 0,
                "bounded_storage_bytes_per_shard": 0,
                "full_storage_bytes_per_shard": 1024,
                "storages": [{
                    "storage_index": 0,
                    "bounded": False,
                    "stride_bytes": 0,
                    "num_slots": 0,
                    "blocks_per_lease": 0,
                    "host_bytes_per_shard": 1024,
                    "free_slots": 0,
                }],
            },
        },
    )
    self.assertEqual(manager.admission_summary(), summary)
    self.assertEqual(manager.num_pools(), 2)
    self.assertEqual(manager.pool_ids_with_tag("kind_a"), [0])
    self.assertEqual(manager.pool_ids_with_tag("kind_b"), [1])
    self.assertEqual(manager.pool_ids_with_tag("missing"), [])

    first_ref = manager.get_block_ref(pool_idx=0, block_id=0)
    first_ref_last = manager.get_block_ref(pool_idx=0, block_id=3)
    second_ref = manager.get_block_ref(pool_idx=1, block_id=2)
    self.assertEqual(first_ref_last["ptr"], first_ref["ptr"] + 3 * 128)
    self.assertEqual(second_ref["ptr"], first_ref["ptr"] + (640 - 32) + 2 * 64)
    self.assertEqual(first_ref["block_stride_bytes"], 128)
    self.assertEqual(second_ref["block_stride_bytes"], 64)
    self.assertEqual(first_ref["tag"], "kind_a")
    self.assertEqual(second_ref["tag"], "kind_b")
    self.assertEqual(second_ref["dtype_tag"], "dtype_b")
    self.assertEqual(manager.pool_spec(1)["base_offset_bytes"], 640)

    with self.assertRaisesRegex(RuntimeError, "host-only manager"):
      manager.d2h_pool_blocks(pool_idx=0, block_ids=[0])
    with self.assertRaisesRegex(RuntimeError, "host-only manager"):
      manager.h2d_pool_blocks(pool_idx=0, block_ids=[0])

  def test_shared_memory_mapping_api(self):
    manager = kv_cache_manager.KVCacheManager.create_host_only_for_testing(
        num_layers=1,
        num_shards=1,
        slice_byte_size=128,
        node_id=7,
        host_blocks=2,
        parallelism=1,
    )
    self.assertFalse(manager.experimental_is_shared_memory_mapped)

    # Map with null address or 0 size raises RuntimeError
    with self.assertRaises(RuntimeError):
      manager.experimental_map_shared_memory(0, 4096)
    with self.assertRaises(RuntimeError):
      manager.experimental_map_shared_memory(4096, 0)

    # Unmap when not mapped raises RuntimeError
    with self.assertRaises(RuntimeError):
      manager.experimental_unmap_shared_memory()

    # Host-only manager has no active PJRT client for DMA mapping
    with self.assertRaisesRegex(RuntimeError, "no active PJRT client"):
      manager.experimental_map_shared_memory(4096, 4096)

    # The binding accepts page_nbytes, and the missing client is reported
    # before the page is validated: page_nbytes=0 would otherwise be rejected.
    with self.assertRaisesRegex(RuntimeError, "no active PJRT client"):
      manager.experimental_map_shared_memory(4096, 4096, page_nbytes=0)


class _FakeDetailsImpl:

  def complete_read(self):
    return (
        ["sent"],
        ["received"],
        ["failed"],
        {"sent": {"completed_ns": 10}, "received": {"completed_ns": 20}},
    )


class _FakeFailureImpl:

  def complete_read(self):
    return (
        [],
        [],
        ["failed"],
        {
            "failed": {
                "completed_ns": 30,
                "elapsed_ns": 20,
                "peer": "10.0.0.2:31000",
                "code": "DEADLINE_EXCEEDED",
                "reason": "Send session timed out",
                "phase": "d2h",
            }
        },
    )


class PollStatsWithDetailsTest(unittest.TestCase):

  def test_attaches_details_to_each_req_id(self):
    manager = object.__new__(kv_cache_manager.KVCacheManager)
    manager._impl = _FakeDetailsImpl()  # pylint: disable=protected-access

    stats = manager.poll_stats_with_details()

    settled = kv_cache_manager.SettledTransfer
    self.assertEqual(stats.sent, [settled("sent", completed_ns=10)])
    self.assertEqual(stats.received, [settled("received", completed_ns=20)])
    self.assertEqual(stats.failed, [settled("failed")])
    self.assertIsNone(stats.failed[0].completed_ns)

  def test_poll_stats_keeps_its_shape(self):
    manager = object.__new__(kv_cache_manager.KVCacheManager)
    manager._impl = _FakeDetailsImpl()  # pylint: disable=protected-access

    self.assertEqual(
        manager.poll_stats(), (["sent"], ["received"], ["failed"])
    )

  def test_failed_transfer_says_why(self):
    manager = object.__new__(kv_cache_manager.KVCacheManager)
    manager._impl = _FakeFailureImpl()  # pylint: disable=protected-access

    [failed] = manager.poll_stats_with_details().failed

    self.assertEqual(failed.req_id, "failed")
    self.assertEqual(failed.completed_ns, 30)
    self.assertEqual(failed.elapsed_ns, 20)
    self.assertEqual(failed.peer, "10.0.0.2:31000")
    self.assertEqual(failed.code, "DEADLINE_EXCEEDED")
    self.assertEqual(failed.reason, "Send session timed out")
    self.assertEqual(failed.phase, "d2h")


class _FakePushImpl:

  def __init__(self):
    self.calls = []

  def push_registered_plan(self, *args):
    self.calls.append(args)


class H2hPushTest(unittest.TestCase):

  def test_forwards_to_the_explicit_destination_push(self):
    manager = object.__new__(kv_cache_manager.KVCacheManager)
    manager._impl = _FakePushImpl()  # pylint: disable=protected-access

    manager.h2h_push(
        peer="10.0.0.2:31000", src_block_ids=(1, 2), dst_block_ids=(7, 8), uuid=9
    )

    self.assertEqual(
        manager._impl.calls,  # pylint: disable=protected-access
        [(9, "10.0.0.2:31000", [1, 2], [7, 8], -1, 1)],
    )

  def test_rejects_empty_or_mismatched_block_lists(self):
    manager = object.__new__(kv_cache_manager.KVCacheManager)
    manager._impl = _FakePushImpl()  # pylint: disable=protected-access

    for src, dst in (([], []), ([1, 2], [7])):
      with self.subTest(src=src, dst=dst):
        with self.assertRaises(ValueError):
          manager.h2h_push(peer="p", src_block_ids=src, dst_block_ids=dst, uuid=1)
    self.assertEqual(manager._impl.calls, [])  # pylint: disable=protected-access

  def test_transfer_failure_is_not_hidden(self):
    class FailingImpl:
      def push_registered_plan(self, *args):
        raise RuntimeError("transfer failed")

    manager = object.__new__(kv_cache_manager.KVCacheManager)
    manager._impl = FailingImpl()  # pylint: disable=protected-access
    with self.assertRaisesRegex(RuntimeError, "transfer failed"):
      manager.h2h_push(peer="p", src_block_ids=[0], dst_block_ids=[1], uuid=1)

  def test_independent_full_and_swa_host_pools(self):
    def make(node_id, layers, block_bytes):
      return kv_cache_manager.KVCacheManager.create_host_only_for_testing(
          num_layers=layers, num_shards=1, slice_byte_size=block_bytes,
          node_id=node_id, host_blocks=4, parallelism=1,
      )

    # Pool geometries need not agree with one another, only with their peer.
    for layers, block_bytes in ((2, 128), (1, 64)):
      src, dst = make(5, layers, block_bytes), make(6, layers, block_bytes)
      for layer in range(layers):
        src.write_block_bytes(layer, 1, bytes([layer + 17]) * block_bytes)
        dst.write_block_bytes(layer, 2, b"\xee" * block_bytes)
      dst.h2h_register_recv(11)
      src.h2h_push(peer=dst.transfer_address, src_block_ids=[1],
                   dst_block_ids=[2], uuid=11)
      dst.h2h_unregister_recv(11)
      for layer in range(layers):
        self.assertEqual(dst.read_block_bytes(layer, 2),
                         bytes([layer + 17]) * block_bytes)

  def test_receive_registration_is_required_and_revoked(self):
    def make(node_id):
      return kv_cache_manager.KVCacheManager.create_host_only_for_testing(
          num_layers=1, num_shards=1, slice_byte_size=128,
          node_id=node_id, host_blocks=4, parallelism=1,
      )
    src, dst = make(7), make(8)
    src.write_block_bytes(0, 0, b"\x01" * 128)
    dst.write_block_bytes(0, 3, b"\xee" * 128)
    dst.h2h_register_recv(13)
    with self.assertRaisesRegex(RuntimeError, "already registered"):
      dst.h2h_register_recv(13)
    dst.h2h_unregister_recv(13)
    with self.assertRaises(RuntimeError):
      src.h2h_push(peer=dst.transfer_address, src_block_ids=[0],
                   dst_block_ids=[3], uuid=13)
    self.assertEqual(dst.read_block_bytes(0, 3), b"\xee" * 128)

  def test_single_layer_does_not_overwrite_other_layers(self):
    def make(node_id):
      return kv_cache_manager.KVCacheManager.create_host_only_for_testing(
          num_layers=2, num_shards=1, slice_byte_size=128,
          node_id=node_id, host_blocks=4, parallelism=1,
      )

    src, dst = make(3), make(4)
    for layer in range(2):
      src.write_block_bytes(layer, 0, bytes([layer + 1]) * 128)
      dst.write_block_bytes(layer, 3, b"\xee" * 128)
    dst.h2h_register_recv(7)
    src.h2h_push(peer=dst.transfer_address, src_block_ids=[0],
                 dst_block_ids=[3], uuid=7, layer_idx=1, parallelism=2)
    dst.h2h_unregister_recv(7)
    self.assertEqual(dst.read_block_bytes(0, 3), b"\xee" * 128)
    self.assertEqual(dst.read_block_bytes(1, 3), b"\x02" * 128)

  def test_pushes_host_blocks_into_named_peer_blocks(self):
    def make(node_id):
      return kv_cache_manager.KVCacheManager.create_host_only_for_testing(
          num_layers=2,
          num_shards=1,
          slice_byte_size=128,
          node_id=node_id,
          host_blocks=4,
          parallelism=1,
      )

    src, dst = make(1), make(2)
    payload = {
        (layer, block): bytes([16 * layer + block]) * 128
        for layer in range(2)
        for block in range(2)
    }
    for (layer, block), data in payload.items():
      src.write_block_bytes(layer, block, data)
    untouched = b"\xee" * 128
    dst.write_block_bytes(0, 1, untouched)

    # Blocks land at the ids the caller names, crossed on purpose.
    dst.h2h_register_recv(5)
    src.h2h_push(
        peer=dst.transfer_address, src_block_ids=[0, 1], dst_block_ids=[3, 2], uuid=5
    )
    dst.h2h_unregister_recv(5)

    for layer in range(2):
      self.assertEqual(dst.read_block_bytes(layer, 3), payload[(layer, 0)])
      self.assertEqual(dst.read_block_bytes(layer, 2), payload[(layer, 1)])
    self.assertEqual(dst.read_block_bytes(0, 1), untouched)


if __name__ == "__main__":
  unittest.main()
