# 运营常量。θ 在 loss.py。

TRANSFER_STATION = (34.66138766837743, 135.4886638680227)

KIND_TO_STATION = "to_station"  # 送机
KIND_FROM_STATION = "from_station"  # 接机

DRIVER_IDLE = "idle"
DRIVER_ENROUTE = "enroute"

AVERAGE_SPEED_KMH = 28.0

PREP_BUFFER_MIN = 5.0  # 发单比出发早这么多
LATE_SLACK_MIN = 10.0  # 线上单没有最晚时刻时，late = 预估时刻 + 这段
AT_TRANSFER_KM = 0.4  # 小于这个距离算已在 T
