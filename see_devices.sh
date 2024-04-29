# export JAX_PLATFORMS=cpu
python -c "import jax
jax.distributed.initialize()
print(len(jax.devices()))"

# [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0), TpuDevice(id=1, process_index=0, coords=(1,0,0), core_on_chip=0), TpuDevice(id=2, process_index=0, coords=(0,1,0), core_on_chip=0), TpuDevice(id=3, process_index=0, coords=(1,1,0), core_on_chip=0), TpuDevice(id=4, process_index=1, coords=(0,2,0), core_on_chip=0), TpuDevice(id=5, process_index=1, coords=(1,2,0), core_on_chip=0), TpuDevice(id=6, process_index=1, coords=(0,3,0), core_on_chip=0), TpuDevice(id=7, process_index=1, coords=(1,3,0), core_on_chip=0), TpuDevice(id=8, process_index=2, coords=(0,0,1), core_on_chip=0), TpuDevice(id=9, process_index=2, coords=(1,0,1), core_on_chip=0), TpuDevice(id=10, process_index=2, coords=(0,1,1), core_on_chip=0), TpuDevice(id=11, process_index=2, coords=(1,1,1), core_on_chip=0), TpuDevice(id=12, process_index=3, coords=(0,2,1), core_on_chip=0), TpuDevice(id=13, process_index=3, coords=(1,2,1), core_on_chip=0), TpuDevice(id=14, process_index=3, coords=(0,3,1), core_on_chip=0), TpuDevice(id=15, process_index=3, coords=(1,3,1), core_on_chip=0), TpuDevice(id=16, process_index=4, coords=(0,0,2), core_on_chip=0), TpuDevice(id=17, process_index=4, coords=(1,0,2), core_on_chip=0), TpuDevice(id=18, process_index=4, coords=(0,1,2), core_on_chip=0), TpuDevice(id=19, process_index=4, coords=(1,1,2), core_on_chip=0), TpuDevice(id=20, process_index=5, coords=(0,2,2), core_on_chip=0), TpuDevice(id=21, process_index=5, coords=(1,2,2), core_on_chip=0), TpuDevice(id=22, process_index=5, coords=(0,3,2), core_on_chip=0), TpuDevice(id=23, process_index=5, coords=(1,3,2), core_on_chip=0), TpuDevice(id=24, process_index=6, coords=(0,0,3), core_on_chip=0), TpuDevice(id=25, process_index=6, coords=(1,0,3), core_on_chip=0), TpuDevice(id=26, process_index=6, coords=(0,1,3), core_on_chip=0), TpuDevice(id=27, process_index=6, coords=(1,1,3), core_on_chip=0), TpuDevice(id=28, process_index=7, coords=(0,2,3), core_on_chip=0), TpuDevice(id=29, process_index=7, coords=(1,2,3), core_on_chip=0), TpuDevice(id=30, process_index=7, coords=(0,3,3), core_on_chip=0), TpuDevice(id=31, process_index=7, coords=(1,3,3), core_on_chip=0)]
# [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0), TpuDevice(id=1, process_index=0, coords=(1,0,0), core_on_chip=0), TpuDevice(id=2, process_index=0, coords=(0,1,0), core_on_chip=0), TpuDevice(id=3, process_index=0, coords=(1,1,0), core_on_chip=0), TpuDevice(id=4, process_index=1, coords=(0,2,0), core_on_chip=0), TpuDevice(id=5, process_index=1, coords=(1,2,0), core_on_chip=0), TpuDevice(id=6, process_index=1, coords=(0,3,0), core_on_chip=0), TpuDevice(id=7, process_index=1, coords=(1,3,0), core_on_chip=0), TpuDevice(id=8, process_index=2, coords=(0,0,1), core_on_chip=0), TpuDevice(id=9, process_index=2, coords=(1,0,1), core_on_chip=0), TpuDevice(id=10, process_index=2, coords=(0,1,1), core_on_chip=0), TpuDevice(id=11, process_index=2, coords=(1,1,1), core_on_chip=0), TpuDevice(id=12, process_index=3, coords=(0,2,1), core_on_chip=0), TpuDevice(id=13, process_index=3, coords=(1,2,1), core_on_chip=0), TpuDevice(id=14, process_index=3, coords=(0,3,1), core_on_chip=0), TpuDevice(id=15, process_index=3, coords=(1,3,1), core_on_chip=0), TpuDevice(id=16, process_index=4, coords=(0,0,2), core_on_chip=0), TpuDevice(id=17, process_index=4, coords=(1,0,2), core_on_chip=0), TpuDevice(id=18, process_index=4, coords=(0,1,2), core_on_chip=0), TpuDevice(id=19, process_index=4, coords=(1,1,2), core_on_chip=0), TpuDevice(id=20, process_index=5, coords=(0,2,2), core_on_chip=0), TpuDevice(id=21, process_index=5, coords=(1,2,2), core_on_chip=0), TpuDevice(id=22, process_index=5, coords=(0,3,2), core_on_chip=0), TpuDevice(id=23, process_index=5, coords=(1,3,2), core_on_chip=0), TpuDevice(id=24, process_index=6, coords=(0,0,3), core_on_chip=0), TpuDevice(id=25, process_index=6, coords=(1,0,3), core_on_chip=0), TpuDevice(id=26, process_index=6, coords=(0,1,3), core_on_chip=0), TpuDevice(id=27, process_index=6, coords=(1,1,3), core_on_chip=0), TpuDevice(id=28, process_index=7, coords=(0,2,3), core_on_chip=0), TpuDevice(id=29, process_index=7, coords=(1,2,3), core_on_chip=0), TpuDevice(id=30, process_index=7, coords=(0,3,3), core_on_chip=0), TpuDevice(id=31, process_index=7, coords=(1,3,3), core_on_chip=0)]
# [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0), TpuDevice(id=1, process_index=0, coords=(1,0,0), core_on_chip=0), TpuDevice(id=2, process_index=0, coords=(0,1,0), core_on_chip=0), TpuDevice(id=3, process_index=0, coords=(1,1,0), core_on_chip=0), TpuDevice(id=4, process_index=1, coords=(0,2,0), core_on_chip=0), TpuDevice(id=5, process_index=1, coords=(1,2,0), core_on_chip=0), TpuDevice(id=6, process_index=1, coords=(0,3,0), core_on_chip=0), TpuDevice(id=7, process_index=1, coords=(1,3,0), core_on_chip=0), TpuDevice(id=8, process_index=2, coords=(0,0,1), core_on_chip=0), TpuDevice(id=9, process_index=2, coords=(1,0,1), core_on_chip=0), TpuDevice(id=10, process_index=2, coords=(0,1,1), core_on_chip=0), TpuDevice(id=11, process_index=2, coords=(1,1,1), core_on_chip=0), TpuDevice(id=12, process_index=3, coords=(0,2,1), core_on_chip=0), TpuDevice(id=13, process_index=3, coords=(1,2,1), core_on_chip=0), TpuDevice(id=14, process_index=3, coords=(0,3,1), core_on_chip=0), TpuDevice(id=15, process_index=3, coords=(1,3,1), core_on_chip=0), TpuDevice(id=16, process_index=4, coords=(0,0,2), core_on_chip=0), TpuDevice(id=17, process_index=4, coords=(1,0,2), core_on_chip=0), TpuDevice(id=18, process_index=4, coords=(0,1,2), core_on_chip=0), TpuDevice(id=19, process_index=4, coords=(1,1,2), core_on_chip=0), TpuDevice(id=20, process_index=5, coords=(0,2,2), core_on_chip=0), TpuDevice(id=21, process_index=5, coords=(1,2,2), core_on_chip=0), TpuDevice(id=22, process_index=5, coords=(0,3,2), core_on_chip=0), TpuDevice(id=23, process_index=5, coords=(1,3,2), core_on_chip=0), TpuDevice(id=24, process_index=6, coords=(0,0,3), core_on_chip=0), TpuDevice(id=25, process_index=6, coords=(1,0,3), core_on_chip=0), TpuDevice(id=26, process_index=6, coords=(0,1,3), core_on_chip=0), TpuDevice(id=27, process_index=6, coords=(1,1,3), core_on_chip=0), TpuDevice(id=28, process_index=7, coords=(0,2,3), core_on_chip=0), TpuDevice(id=29, process_index=7, coords=(1,2,3), core_on_chip=0), TpuDevice(id=30, process_index=7, coords=(0,3,3), core_on_chip=0), TpuDevice(id=31, process_index=7, coords=(1,3,3), core_on_chip=0)]
# [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0), TpuDevice(id=1, process_index=0, coords=(1,0,0), core_on_chip=0), TpuDevice(id=2, process_index=0, coords=(0,1,0), core_on_chip=0), TpuDevice(id=3, process_index=0, coords=(1,1,0), core_on_chip=0), TpuDevice(id=4, process_index=1, coords=(0,2,0), core_on_chip=0), TpuDevice(id=5, process_index=1, coords=(1,2,0), core_on_chip=0), TpuDevice(id=6, process_index=1, coords=(0,3,0), core_on_chip=0), TpuDevice(id=7, process_index=1, coords=(1,3,0), core_on_chip=0), TpuDevice(id=8, process_index=2, coords=(0,0,1), core_on_chip=0), TpuDevice(id=9, process_index=2, coords=(1,0,1), core_on_chip=0), TpuDevice(id=10, process_index=2, coords=(0,1,1), core_on_chip=0), TpuDevice(id=11, process_index=2, coords=(1,1,1), core_on_chip=0), TpuDevice(id=12, process_index=3, coords=(0,2,1), core_on_chip=0), TpuDevice(id=13, process_index=3, coords=(1,2,1), core_on_chip=0), TpuDevice(id=14, process_index=3, coords=(0,3,1), core_on_chip=0), TpuDevice(id=15, process_index=3, coords=(1,3,1), core_on_chip=0), TpuDevice(id=16, process_index=4, coords=(0,0,2), core_on_chip=0), TpuDevice(id=17, process_index=4, coords=(1,0,2), core_on_chip=0), TpuDevice(id=18, process_index=4, coords=(0,1,2), core_on_chip=0), TpuDevice(id=19, process_index=4, coords=(1,1,2), core_on_chip=0), TpuDevice(id=20, process_index=5, coords=(0,2,2), core_on_chip=0), TpuDevice(id=21, process_index=5, coords=(1,2,2), core_on_chip=0), TpuDevice(id=22, process_index=5, coords=(0,3,2), core_on_chip=0), TpuDevice(id=23, process_index=5, coords=(1,3,2), core_on_chip=0), TpuDevice(id=24, process_index=6, coords=(0,0,3), core_on_chip=0), TpuDevice(id=25, process_index=6, coords=(1,0,3), core_on_chip=0), TpuDevice(id=26, process_index=6, coords=(0,1,3), core_on_chip=0), TpuDevice(id=27, process_index=6, coords=(1,1,3), core_on_chip=0), TpuDevice(id=28, process_index=7, coords=(0,2,3), core_on_chip=0), TpuDevice(id=29, process_index=7, coords=(1,2,3), core_on_chip=0), TpuDevice(id=30, process_index=7, coords=(0,3,3), core_on_chip=0), TpuDevice(id=31, process_index=7, coords=(1,3,3), core_on_chip=0)]
# [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0), TpuDevice(id=1, process_index=0, coords=(1,0,0), core_on_chip=0), TpuDevice(id=2, process_index=0, coords=(0,1,0), core_on_chip=0), TpuDevice(id=3, process_index=0, coords=(1,1,0), core_on_chip=0), TpuDevice(id=4, process_index=1, coords=(0,2,0), core_on_chip=0), TpuDevice(id=5, process_index=1, coords=(1,2,0), core_on_chip=0), TpuDevice(id=6, process_index=1, coords=(0,3,0), core_on_chip=0), TpuDevice(id=7, process_index=1, coords=(1,3,0), core_on_chip=0), TpuDevice(id=8, process_index=2, coords=(0,0,1), core_on_chip=0), TpuDevice(id=9, process_index=2, coords=(1,0,1), core_on_chip=0), TpuDevice(id=10, process_index=2, coords=(0,1,1), core_on_chip=0), TpuDevice(id=11, process_index=2, coords=(1,1,1), core_on_chip=0), TpuDevice(id=12, process_index=3, coords=(0,2,1), core_on_chip=0), TpuDevice(id=13, process_index=3, coords=(1,2,1), core_on_chip=0), TpuDevice(id=14, process_index=3, coords=(0,3,1), core_on_chip=0), TpuDevice(id=15, process_index=3, coords=(1,3,1), core_on_chip=0), TpuDevice(id=16, process_index=4, coords=(0,0,2), core_on_chip=0), TpuDevice(id=17, process_index=4, coords=(1,0,2), core_on_chip=0), TpuDevice(id=18, process_index=4, coords=(0,1,2), core_on_chip=0), TpuDevice(id=19, process_index=4, coords=(1,1,2), core_on_chip=0), TpuDevice(id=20, process_index=5, coords=(0,2,2), core_on_chip=0), TpuDevice(id=21, process_index=5, coords=(1,2,2), core_on_chip=0), TpuDevice(id=22, process_index=5, coords=(0,3,2), core_on_chip=0), TpuDevice(id=23, process_index=5, coords=(1,3,2), core_on_chip=0), TpuDevice(id=24, process_index=6, coords=(0,0,3), core_on_chip=0), TpuDevice(id=25, process_index=6, coords=(1,0,3), core_on_chip=0), TpuDevice(id=26, process_index=6, coords=(0,1,3), core_on_chip=0), TpuDevice(id=27, process_index=6, coords=(1,1,3), core_on_chip=0), TpuDevice(id=28, process_index=7, coords=(0,2,3), core_on_chip=0), TpuDevice(id=29, process_index=7, coords=(1,2,3), core_on_chip=0), TpuDevice(id=30, process_index=7, coords=(0,3,3), core_on_chip=0), TpuDevice(id=31, process_index=7, coords=(1,3,3), core_on_chip=0)]
# [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0), TpuDevice(id=1, process_index=0, coords=(1,0,0), core_on_chip=0), TpuDevice(id=2, process_index=0, coords=(0,1,0), core_on_chip=0), TpuDevice(id=3, process_index=0, coords=(1,1,0), core_on_chip=0), TpuDevice(id=4, process_index=1, coords=(0,2,0), core_on_chip=0), TpuDevice(id=5, process_index=1, coords=(1,2,0), core_on_chip=0), TpuDevice(id=6, process_index=1, coords=(0,3,0), core_on_chip=0), TpuDevice(id=7, process_index=1, coords=(1,3,0), core_on_chip=0), TpuDevice(id=8, process_index=2, coords=(0,0,1), core_on_chip=0), TpuDevice(id=9, process_index=2, coords=(1,0,1), core_on_chip=0), TpuDevice(id=10, process_index=2, coords=(0,1,1), core_on_chip=0), TpuDevice(id=11, process_index=2, coords=(1,1,1), core_on_chip=0), TpuDevice(id=12, process_index=3, coords=(0,2,1), core_on_chip=0), TpuDevice(id=13, process_index=3, coords=(1,2,1), core_on_chip=0), TpuDevice(id=14, process_index=3, coords=(0,3,1), core_on_chip=0), TpuDevice(id=15, process_index=3, coords=(1,3,1), core_on_chip=0), TpuDevice(id=16, process_index=4, coords=(0,0,2), core_on_chip=0), TpuDevice(id=17, process_index=4, coords=(1,0,2), core_on_chip=0), TpuDevice(id=18, process_index=4, coords=(0,1,2), core_on_chip=0), TpuDevice(id=19, process_index=4, coords=(1,1,2), core_on_chip=0), TpuDevice(id=20, process_index=5, coords=(0,2,2), core_on_chip=0), TpuDevice(id=21, process_index=5, coords=(1,2,2), core_on_chip=0), TpuDevice(id=22, process_index=5, coords=(0,3,2), core_on_chip=0), TpuDevice(id=23, process_index=5, coords=(1,3,2), core_on_chip=0), TpuDevice(id=24, process_index=6, coords=(0,0,3), core_on_chip=0), TpuDevice(id=25, process_index=6, coords=(1,0,3), core_on_chip=0), TpuDevice(id=26, process_index=6, coords=(0,1,3), core_on_chip=0), TpuDevice(id=27, process_index=6, coords=(1,1,3), core_on_chip=0), TpuDevice(id=28, process_index=7, coords=(0,2,3), core_on_chip=0), TpuDevice(id=29, process_index=7, coords=(1,2,3), core_on_chip=0), TpuDevice(id=30, process_index=7, coords=(0,3,3), core_on_chip=0), TpuDevice(id=31, process_index=7, coords=(1,3,3), core_on_chip=0)]
# [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0), TpuDevice(id=1, process_index=0, coords=(1,0,0), core_on_chip=0), TpuDevice(id=2, process_index=0, coords=(0,1,0), core_on_chip=0), TpuDevice(id=3, process_index=0, coords=(1,1,0), core_on_chip=0), TpuDevice(id=4, process_index=1, coords=(0,2,0), core_on_chip=0), TpuDevice(id=5, process_index=1, coords=(1,2,0), core_on_chip=0), TpuDevice(id=6, process_index=1, coords=(0,3,0), core_on_chip=0), TpuDevice(id=7, process_index=1, coords=(1,3,0), core_on_chip=0), TpuDevice(id=8, process_index=2, coords=(0,0,1), core_on_chip=0), TpuDevice(id=9, process_index=2, coords=(1,0,1), core_on_chip=0), TpuDevice(id=10, process_index=2, coords=(0,1,1), core_on_chip=0), TpuDevice(id=11, process_index=2, coords=(1,1,1), core_on_chip=0), TpuDevice(id=12, process_index=3, coords=(0,2,1), core_on_chip=0), TpuDevice(id=13, process_index=3, coords=(1,2,1), core_on_chip=0), TpuDevice(id=14, process_index=3, coords=(0,3,1), core_on_chip=0), TpuDevice(id=15, process_index=3, coords=(1,3,1), core_on_chip=0), TpuDevice(id=16, process_index=4, coords=(0,0,2), core_on_chip=0), TpuDevice(id=17, process_index=4, coords=(1,0,2), core_on_chip=0), TpuDevice(id=18, process_index=4, coords=(0,1,2), core_on_chip=0), TpuDevice(id=19, process_index=4, coords=(1,1,2), core_on_chip=0), TpuDevice(id=20, process_index=5, coords=(0,2,2), core_on_chip=0), TpuDevice(id=21, process_index=5, coords=(1,2,2), core_on_chip=0), TpuDevice(id=22, process_index=5, coords=(0,3,2), core_on_chip=0), TpuDevice(id=23, process_index=5, coords=(1,3,2), core_on_chip=0), TpuDevice(id=24, process_index=6, coords=(0,0,3), core_on_chip=0), TpuDevice(id=25, process_index=6, coords=(1,0,3), core_on_chip=0), TpuDevice(id=26, process_index=6, coords=(0,1,3), core_on_chip=0), TpuDevice(id=27, process_index=6, coords=(1,1,3), core_on_chip=0), TpuDevice(id=28, process_index=7, coords=(0,2,3), core_on_chip=0), TpuDevice(id=29, process_index=7, coords=(1,2,3), core_on_chip=0), TpuDevice(id=30, process_index=7, coords=(0,3,3), core_on_chip=0), TpuDevice(id=31, process_index=7, coords=(1,3,3), core_on_chip=0)]
# [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0), TpuDevice(id=1, process_index=0, coords=(1,0,0), core_on_chip=0), TpuDevice(id=2, process_index=0, coords=(0,1,0), core_on_chip=0), TpuDevice(id=3, process_index=0, coords=(1,1,0), core_on_chip=0), TpuDevice(id=4, process_index=1, coords=(0,2,0), core_on_chip=0), TpuDevice(id=5, process_index=1, coords=(1,2,0), core_on_chip=0), TpuDevice(id=6, process_index=1, coords=(0,3,0), core_on_chip=0), TpuDevice(id=7, process_index=1, coords=(1,3,0), core_on_chip=0), TpuDevice(id=8, process_index=2, coords=(0,0,1), core_on_chip=0), TpuDevice(id=9, process_index=2, coords=(1,0,1), core_on_chip=0), TpuDevice(id=10, process_index=2, coords=(0,1,1), core_on_chip=0), TpuDevice(id=11, process_index=2, coords=(1,1,1), core_on_chip=0), TpuDevice(id=12, process_index=3, coords=(0,2,1), core_on_chip=0), TpuDevice(id=13, process_index=3, coords=(1,2,1), core_on_chip=0), TpuDevice(id=14, process_index=3, coords=(0,3,1), core_on_chip=0), TpuDevice(id=15, process_index=3, coords=(1,3,1), core_on_chip=0), TpuDevice(id=16, process_index=4, coords=(0,0,2), core_on_chip=0), TpuDevice(id=17, process_index=4, coords=(1,0,2), core_on_chip=0), TpuDevice(id=18, process_index=4, coords=(0,1,2), core_on_chip=0), TpuDevice(id=19, process_index=4, coords=(1,1,2), core_on_chip=0), TpuDevice(id=20, process_index=5, coords=(0,2,2), core_on_chip=0), TpuDevice(id=21, process_index=5, coords=(1,2,2), core_on_chip=0), TpuDevice(id=22, process_index=5, coords=(0,3,2), core_on_chip=0), TpuDevice(id=23, process_index=5, coords=(1,3,2), core_on_chip=0), TpuDevice(id=24, process_index=6, coords=(0,0,3), core_on_chip=0), TpuDevice(id=25, process_index=6, coords=(1,0,3), core_on_chip=0), TpuDevice(id=26, process_index=6, coords=(0,1,3), core_on_chip=0), TpuDevice(id=27, process_index=6, coords=(1,1,3), core_on_chip=0), TpuDevice(id=28, process_index=7, coords=(0,2,3), core_on_chip=0), TpuDevice(id=29, process_index=7, coords=(1,2,3), core_on_chip=0), TpuDevice(id=30, process_index=7, coords=(0,3,3), core_on_chip=0), TpuDevice(id=31, process_index=7, coords=(1,3,3), core_on_chip=0)]