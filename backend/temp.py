CATEGORY_DICT = {
    "장비 상태 및 기본 정보" :
    """
    endpoint 형식:
        • 일반 정보: /machine/{leaf_node}
        • NC 메모리 정보: /machine/ncMemory/{leaf_node}

        === 일반 장비 정보 ===
        • cncModel - 해당 장비에 탑재된 NC의 모델명(STRING)
        • numberOfChannels - 장비에서 사용 가능한 채널(계통)의 개수(INTEGER)
        • cncVendor - NC 제조사 코드 (1: FANUC, 2: SIEMENS, 3:CSCAM, 4: MITSUBISHI, 5: KCNC)(INTEGER)
        • ncLinkState - NC와의 통신 가능 여부(BOOLEAN)
        • currentAccessLevel - 프로그램/디렉토리 접근 권한 수준 (SIEMENS 전용). 1: 제조자, 2: 서비스, 3: 사용자, 4: 프로그래머(키 스위치 3), 5: 공인 전문가(키 스위치 2), 6: 숙련된 전문가(키 스위치 1), 7: 준 숙련 전문가(키 스위치 0)(INTEGER)
        • basicLengthUnit - 장비가 사용하는 기본 길이 단위 (0: Metric, 1: Inches, 4: user Define 등)(INTEGER)
        • machinePowerOnTime - 장비의 전원이 켜진 시간 (단위: 분)(REAL)
        • currentCncTime - 장비에 설정된 현재 시각 (형식: yyyy-MM-ddTHH:mm:ss)(STRING)
        • machineType - 장비의 타입 (0: 알 수 없음, 1: Milling, 2: Lathe)(INTEGER)

        === NC 메모리 정보 ===
        • ncMemory/totalCapacity - NC 메모리의 전체 용량 (단위: byte)(REAL)
        • ncMemory/usedCapacity - 사용 중인 NC 메모리 용량 (단위: byte)(REAL)
        • ncMemory/freeCapacity - NC 메모리의 남은 용량 (단위: byte)(REAL)
        • ncMemory/rootPath - NC 메모리의 기본(루트) 경로(STRING)

        예시:
        - endpoint="/machine/cncModel"
        - endpoint="/machine/ncMemory/freeCapacity"
        - params= {"machine": 1}
    """
    ,
    "계통 별 기록되는 채널의 상태 정보" : 
    """
    endpoint 형식: /machine/channel/{leaf_node}
    
        사용 가능한 leaf_node:
        • channelEnabled  - 해당 채널의 사용 가능 여부(BOOLEAN)
        • toolAreaNumber  - 해당 채널에서 사용 가능한 공구 영역의 식별 번호. 단계통 장비의 경우 디폴트로 1. FANUC에서는 공구 영역과 계통이 동일하기 때문에 channel과 toolArea가 같은 개념으로 사용. SIEMENS의 공구 영역의 개수는 계통 수와 동등하며, 공구 영역과 계통 간 1:다 관계가 성립.(INTEGER)  
        • numberOfAxes  - 해당 채널에서 사용 가능한 축의 개수(INTEGER)
        • numberOfSpindles  - 해당 채널에서 사용 가능한 스핀들의 개수.(INTEGER)
        • alarmStatus   - 채널의 알람 상태. 0: no alarm, 1: alarm, 2: alarm without stop, 3: alarm with stop, 4: Battery low, 5: FAN, 6: PS warning, 7: FSSB waring, 8: Insulate warning, 9: Encoder warning 10: PMC alarm(INTEGER)
        • numberOfAlarms   - 해당 채널에서 발생한 알람의 총 개수(INTEGER)  
        • operateMode   - 공작기계의 운전 모드 (0: JOG, 1: MDI, 2: MEMORY(AUTO), 3: ZRN, 4: MPG, 5: **** 6: EDIT, 7: HANDLE, 8: Teach in JOG, 9: Teach in HANDLE 10: INC·feed, 11: REFERENCE, 12: REMOTE, 13: JOG-REPOS, 14: MDI-REF.POINT, 15: MDI-TEACH IN, 16: MDI-TECH IN-REF.POINT, 17: AUTO-TECH IN-REF.POINT 18: STEP, 19:RAPID, 20: TAPE, 21: AUTO-TEACH IN-JOG, 22: JOG-REF)(INTEGER)
        • numberOfWorkOffsets   - 사용 가능한 공작물 좌표계의 개수(INTEGER)
        • ncState   - CNC의 작동 상태 (0: Reset, 1: Stop, 2: Hold, 3: Start, 4: MSTR, 5: Interrupted, 6: Pause)(INTEGER)
        • motionStatus   - 장비의 현재 모션 상태 (1: Motion, 2: Dwell, 3: Wait)(INTEGER)  
        • emergencyStatus   - 상태 여부 (0: Not emergency, 1: Emergency, 2: Reset, 3: Wait)(INTEGER)
   

        예시: endpoint="/machine/channel/channelEnabled", params={"machine": 1, "channel": 1}
    """
    ,
    "축 별 상태 정보" : 
    """
    endpoint 형식: 
        • 일반 정보: /machine/channel/axis/{leaf_node}
        • 전력 정보: /machine/channel/axis/axisPower/{leaf_node}
            
         === 일반 축 정보 leaf_node ===
        • machinePosition - 기계 좌표계 기준 현재 위치(REAL)
        • workPosition - 공작물 좌표계 기준 현재 위치(REAL)
        • distanceToGo - 지령 위치까지 남은 이동 거리(REAL)
        • relativePosition - 상대 좌표계 기준 현재 위치(REAL)
        • axisName - 절대 좌표계의 축 이름(STRING)
        • relativeAxisName - 상대 좌표계의 축 이름 (FANUC 전용)(STRING)
        • axisLoad - 축에 걸리는 부하(REAL)
        • axisFeed - 현재 축의 이송 속도(REAL)
        • axisLimitPlus - '+' 방향 최대 이동 한계값(REAL)
        • axisLimitMinus - '-' 방향 최대 이동 한계값(REAL)
        • workAreaLimitPlus - 작업 금지 영역 '+' 방향 한계값(REAL)
        • workAreaLimitMinus - 작업 금지 영역 '-' 방향 한계값(REAL)
        • workAreaLimitPlusEnabled - 작업 금지 영역 '+' 방향 활성화 여부(BOOLEAN)
        • workAreaLimitMinusEnabled - 작업 금지 영역 '-' 방향 활성화 여부(BOOLEAN)
        • axisEnabled - 해당 축의 사용 가능 여부(BOOLEAN)
        • interlockEnabled - 해당 축의 인터락 상태 여부(BOOLEAN)
        • constantSurfaceSpeedControlEnabled - 주속 일정 제어(CSS) 활성화 여부(BOOLEAN)
        • axisCurrent - 해당 축의 전류 정보(REAL)
        • machineOrigin - 기계 원점 좌표값(REAL)
        • axisTemperature - 해당 축의 온도 정보(REAL)
        
        === 축 전력 정보 ===  
        • axisPower/actualPowerConsumption - 실 소비 전력 적산값(REAL)
        • axisPower/powerConsumption - 소비 전력 적산값(REAL)
        • axisPower/regeneratedPower - 회생 전력 적산값(REAL)
    
        예시: endpoint="/machine/channel/axis/axisLoad", params={"machine": 1, "channel": 1, "axis": 1}
    """,
    "스핀들 별 상태 정보" : 
    """
    endpoint 형식:
        • 일반 정보: /machine/channel/spindle/{leaf_node}
        • RPM 정보: /machine/channel/spindle/rpm/{leaf_node}
        • 전력 정보: /machine/channel/spindle/spindlePower/{leaf_node}

        === 일반 스핀들 정보 ===
        • spindleLoad - 스핀들에 걸리는 부하(REAL)
        • spindleOverride - 스핀들 속도 오버라이드 비율(REAL)
        • spindleLimit - 최대 회전 속도 한계값(REAL)
        • spindleEnabled - 해당 스핀들의 사용 가능 여부(BOOLEAN)
        • spindleCurrent - 해당 스핀들의 전류 정보(REAL)
        • spindleTemperature - 해당 스핀들의 온도 정보(REAL)

        === 스핀들 RPM 정보 ===
        • rpm/commandedSpeed - 지령된 스핀들 회전 속도(REAL)
        • rpm/actualSpeed - 실제 측정된 스핀들 회전 속도(REAL)
        • rpm/speedUnit - 속도 단위 (0: mm/min, 1: inch/min, 2: rpm, 3: mm/rev, 4: inch/rev 등)(INTEGER)

        === 스핀들 전력 정보 ===
        • spindlePower/actualPowerConsumption - 실 소비 전력의 적산값(REAL)
        • spindlePower/powerConsumption - 소비 전력의 적산값(REAL)
        • spindlePower/regeneratedPower - 회생 전력의 적산값(REAL)

        예시:
        - endpoint="/machine/channel/spindle/spindleLoad"
        - endpoint="/machine/channel/spindle/rpm/actualSpeed"
        - endpoint="/machine/channel/spindle/spindlePower/powerConsumption"
        - params={"machine": 1, "channel": 1, "spindle": 1}
    """,
    "축 이송 정보" : 
    """
    endpoint 형식:
        • 오버라이드 정보: /machine/channel/feed/{leaf_node}
        • 이송 속도 정보: /machine/channel/feed/feedRate/{leaf_node}

        === 이송 오버라이드 정보 ===
        • feedOverride - 가공 이송 속도 오버라이드 비율(REAL)
        • rapidOverride - 급속 이송 속도 오버라이드 비율(REAL)

        === 이송 속도 정보 ===
        • feedRate/commandedSpeed - 지령된 이송 속도(REAL)
        • feedRate/actualSpeed - 실제 측정된 이송 속도(REAL)
        • feedRate/speedUnit - 속도 단위  (0: mm/min, 1: inch/min, 2: rpm, 3: mm/rev, 4: inch/rev 등)(INTEGER)

        예시:
        - endpoint="/machine/channel/feed/feedOverride"
        - endpoint="/machine/channel/feed/feedRate/actualSpeed"
        - params={"machine": 1, "channel": 1}
    """,
    "가공 작업의 진척 상태 정보" : 
    """
    endpoint 형식:
        • 가공 수량 정보: /machine/channel/workStatus/workCounter/{leaf_node}
        • 가공 시간 정보: /machine/channel/workStatus/machiningTime/{leaf_node}


        === 가공 수량 정보 ===
        • workCounter/currentWorkCounter - 현재까지 가공한 수량(INTEGER)
        • workCounter/targetWorkCounter - 목표 가공 수량(INTEGER)
        • workCounter/totalWorkCounter - 총 가공 수량(INTEGER)

        === 가공 시간 정보 ===
        • machiningTime/processingMachiningTime - 현재 가공이 진행된 시간 (단위: 초)(REAL)
        • machiningTime/estimatedMachiningTime - 예상 남은 가공 완료 시간 (SIEMENS 전용)(REAL)
        • machiningTime/machineOperationTime - 자동 운전 모드에서의 총 운전 시간 (단위: 초)(REAL)
        • machiningTime/actualCuttingTime - 실제 총 절삭 시간 (단위: 초)(REAL)

        예시:
        - endpoint="/machine/channel/workStatus/workCounter/currentWorkCounter"
        - params={"machine": 1, "channel": 1, "workStatus": 1}
        - endpoint="/machine/channel/workStatus/machiningTime/processingMachiningTime"
        - params={"machine": 1, "channel": 1, "workStatus": 1}
    """,
    "현재 활성된 공구의 상세 정보" :  
    """
    endpoint 형식:
        • 일반 정보: /machine/channel/activeTool/{leaf_node}
        • 공구 날 정보: /machine/channel/activeTool/toolEdge/{leaf_node}
        • 공구 수명 정보: /machine/channel/activeTool/toolEdge/toolLife/{leaf_node}

        === 일반 공구 정보 ===
        • locationNumber - 공구가 매거진에 탑재된 위치 번호(INTEGER)
        • toolName - 공구 이름(STRING)
        • toolNumber - 공구 식별 번호 (T 코드)(INTEGER)
        • numberOfEdges - 공구 날의 총 개수(INTEGER)
        • toolEnabled - 공구 영역 등록 및 매거진 탑재 여부 0: 공구 영역 미등록, 매거진 미탑재 상태, 1: 공구 영역 등록, 매거진 미탑재 상태, 2: 공구 영역 등록, 매거진 탑재 상태(INTEGER)
        • magazineNumber - 공구가 탑재된 매거진 번호(INTEGER)
        • sisterToolNumber - 할당된 대체 공구 번호(INTEGER)
        • toolLifeUnit - 공구 수명 측정 단위 기준.  0: no unit, 1: time, 2: count, 4: wear, 5: count(장착) 6: count(사용), 8: offset (INTEGER)
        • toolGroupNumber - 공구가 참조된 공구 그룹 번호 리스트(INTEGER)
        • toolUseOrderNumber - 그룹 내 공구 사용 순서 (FANUC 전용)(INTEGER)
        • toolStatus - 공구의 사용 상태 0 : Not enabled, 1 : Active tool, 2 : Enabled, 4 : Disabled, 8 : Measured, 9: 미사용 공구, 10 : 정상 수명 공구, 11 : Tool data is available (using), 12 : This tool is registered (available), 13 : This tool has expired, 14 : This tool was skipped, 16 : Prewarning limit reached , 32 : Tool being changed , 64 : Fixed location coded, 128 : Tool was in use , 256 : Tool is in the buffer magazine with transport order, 512 : Ignore disabled state of tool, 1024 : Tool must be unloaded, 2048 : Tool must be loaded, 4096 : Tool is a master tool, 8192 : Reserved, 16384 : Tool is marked for 1:1 exchange, 32768 : Tool is being used as a manual tool (INTEGER)

        === 공구 날(Edge) 정보 ===
        • toolEdge/edgeNumber - 공구 날 식별 번호(INTEGER)
        • toolEdge/toolType - 공구 유형 0: Not defined, 10: General-purpose tool, 11: Threading tool (Siemens에서는 540), 12: Grooving tool, 13: Round-nose tool, 14: Point nose straight tool, 15: Versatile tool, 20: Drill, 21: Counter sink tool, 22: Flat end mill, 23: Ball end mill, 24: Tap (Siemens에서는 240), 25: Reamer, 26: Boring tool, 27: Face mill, 50: Radius end mill, 51: 면취, 52: 선삭, 53: 홈삽입, 54: 나사절삭, 55: 선삭드릴, 56: 선삭탭, 100: Milling tool, 110: Ball nose end mill, 111: Conical ball end, 120: End mill, 121: End mill corner rounding, 130: Angle head cutter, 131: Corner rounding angle head cutter, 140: Facing tool, 145: Thread cutter, 150: Side mill, 151: Saw, 155: Bevelled cutter, 156: Bevelled cutter corner, 157: Tap. die-sink. cutter, 160: Drill&thread cut., 200: Twist drill, 205: Solid drill, 210: Boring bar, 220: Center drill, 230: Countersink, 231: Counterbore, 240: Tap, 241: Fine tap, 242: Tap, Whitworth, 250: Reamer, 500: Roughing tool, 510: Finishing tool, 520: Plunge cutter, 530: Cutting tool, 540: Threading tool, 550: Button tool, 560: Rotary drill, 580: 3D turning probe, 585: Calibrating tool, 700: Slotting saw, 710: 3D probe, 711: Edge finder, 712: Mono probe, 713: L probe, 714: Star probe, 725: Calibrating tool, 730: Stop, 731: Mandrel, 732: Steady rest, 900: Auxiliary tools(INTEGER)
        • toolEdge/lengthOffsetNumber - 공구 길이 보정 식별 번호(INTEGER)
        • toolEdge/geoLengthOffset - 공구 길이 X 보정값(REAL)
        • toolEdge/wearLengthOffset - 공구 길이 X 마모 보정값(REAL)
        • toolEdge/radiusOffsetNumber - 공구 반경 보정 식별 번호(INTEGER)
        • toolEdge/geoRadiusOffset - 공구 반경 보정값(REAL)
        • toolEdge/wearRadiusOffset - 공구 반경 마모 보정값(REAL)
        • toolEdge/edgeEnabled - 공구 날 사용 가능 여부(BOOLEAN)
        • toolEdge/geoLengthOffsetZ - 공구 길이 Z 보정값(REAL)
        • toolEdge/wearLengthOffsetZ - 공구 길이 Z 마모 보정값(REAL)
        • toolEdge/geoLengthOffsetY - 공구 길이 Y 보정값(REAL)
        • toolEdge/wearLengthOffsetY - 공구 길이 Y 마모 보정값(REAL)
        • toolEdge/geoOffsetNumber - 길이 X,Z, 반경의 식별 번호(INTEGER)
        • toolEdge/wearOffsetNumber - 길이 X,Z, 반경 마모값의 식별 번호(INTEGER)
        • toolEdge/cuttingEdgePosition - 공구 인선 방향(INTEGER)
        • toolEdge/tipAngle - 공구의 팁 각도(REAL)
        • toolEdge/holderAngle - 공구 홀더 각도(REAL)
        • toolEdge/insertAngle - 공구 인서트 각도(REAL)
        • toolEdge/insertWidth - 인선 너비 (SIEMENS 전용)(REAL)
        • toolEdge/insertLength - 인선 길이 (SIEMENS 전용)(REAL)
        • toolEdge/referenceDirectionHolderAngle - 홀더 각도 참조 방향 (SIEMENS 전용)(REAL)
        • toolEdge/directionOfSpindleRotation - 스핀들 회전 방향  0: 회전 없음, 1: 시계 방향, 2: 반시계 방향(SIEMENS 전용)(INTEGER)
        • toolEdge/numberOfTeeth - 공구 날 개수 (SIEMENS 전용)(INTEGER)
        
        === 공구 수명 정보 ===
        • toolEdge/toolLife/maxToolLife - 최대 공구 수명(REAL)
        • toolEdge/toolLife/restToolLife - 잔여 공구 수명(REAL)
        • toolEdge/toolLife/toolLifeCount - 현재 공구 사용량(REAL)
        • toolEdge/toolLife/toolLifeAlarm - 공구 수명 도달 경고 설정값 (SIEMENS 전용)(REAL)

        예시:
        - params = {"machine": 1, "channel": 1}
        - endpoint = "/machine/channel/activeTool/toolNumber"
        
        - params = {"machine": 1, "channel": 1}
        - endpoint = "/machine/channel/activeTool/toolEdge/geoLengthOffset"

        - params = {"machine": 1, "channel": 1}
        - endpoint = "/machine/channel/activeTool/toolEdge/toolLife/restToolLife"
    """,
    "현재 실행 중인 NC 프로그램의 상태 정보" : 
    """
    endpoint 형식:
        • 일반 정보: /machine/channel/currentProgram/{leaf_node}
        • 모달 정보: /machine/channel/currentProgram/modal/{leaf_node}
        • 실행 블록 정보: /machine/channel/currentProgram/overallBlock/{leaf_node}
        • 중단점 정보: /machine/channel/currentProgram/interruptBlock/{leaf_node}
        • 좌표계 오프셋 정보: /machine/channel/currentProgram/currentTotalWorkOffset/{leaf_node}
        • 현재 파일 정보: /machine/channel/currentProgram/currentFile/{leaf_node}
        • 메인 파일 정보: /machine/channel/currentProgram/mainFile/{leaf_node}
        • 제어 옵션 정보: /machine/channel/currentProgram/controlOption/{leaf_node}


        === 일반 프로그램 정보 ===
        • sequenceNumber - 현재 실행 중인 시퀀스 번호(N 코드)(INTEGER)
        • currentBlockCounter - 실행 중인 블록 카운터(INTEGER)
        • lastBlock - 이전 블록 정보(STRING)
        • currentBlock - 현재 실행 중인 프로그램 블록 내용(STRING)
        • nextBlock - 다음 블록 정보(STRING)
        • activePartProgram - 실행 중인 프로그램 블록 정보(최대 200자)(STRING)
        • programMode - 프로그램 실행 모드  0: Reset, 1: Stop, 2: Hold, 3: Start(Active)(run), 4: MSTR, 5: Interrupted, 6: Pause, 7: Waiting (INTEGER)
        • currentWorkOffsetIndex - 현재 공작물 좌표계의 G 코드 인덱스(INTEGER)
        • currentWorkOffsetCode - 현재 공작물 좌표계의 G 코드 문자열(STRING)
        • currentDepthLevel - 현재 프로그램의 레벨 (메인, 서브루틴 등)(INTEGER)

        === G 코드 모달 정보 ===
        • modal/modalIndex - G 코드 인덱스(INTEGER)
        • modal/modalCode - G 코드 문자열(STRING)

        === 실행 블록 정보 (SIEMENS) ===
        • overallBlock/blockCounter - 블록 카운터(INTEGER)
        • overallBlock/programName - 프로그램 이름(STRING)

        === 중단점 블록 정보 (SIEMENS) ===
        • interruptBlock/depthLevel - 중단점 블록의 프로그램 레벨 (INTEGER)
        • interruptBlock/blockCounter - 중단점 블록의 카운터(INTEGER)
        • interruptBlock/programName - 중단점 블록의 프로그램 이름 (STRING)
        • interruptBlock/blockData - 중단점 블록 데이터 (STRING)
        • interruptBlock/searchType - 중단점 검색 유형 (INTEGER)
        • interruptBlock/mainProgramName - 중단점의 메인 프로그램 이름 (STRING)

        === 공작물 좌표계 오프셋 정보 ===
        • currentTotalWorkOffset/workOffsetIndex - G 코드 인덱스(INTEGER)
        • currentTotalWorkOffset/workOffsetValue - 축별 총 오프셋 값 (REAL)
        • currentTotalWorkOffset/workOffsetRotation - 축별 총 회전 오프셋 값 (REAL)
        • currentTotalWorkOffset/workOffsetScalingFactor - 축별 총 스케일링 값 (REAL)
        • currentTotalWorkOffset/workOffsetMirroringEnabled - 축별 미러링 활성화 여부 (BOOLEAN)

        === 현재 실행 파일 정보 ===
        • currentFile/programName - 파일명(STRING)
        • currentFile/programPath - 파일 경로(STRING)
        • currentFile/programSize - 파일 크기 (byte)(REAL)
        • currentFile/programDate - 파일 생성 날짜(STRING)
        • currentFile/programNameWithPath - 경로를 포함한 전체 파일명(STRING)

        === 메인 프로그램 파일 정보 ===
        • mainFile/programName - 파일명(STRING)
        • mainFile/programPath - 파일 경로(STRING)
        • mainFile/programSize - 파일 크기 (byte)(REAL)
        • mainFile/programDate - 파일 생성 날짜(STRING)
        • mainFile/programNameWithPath - 경로를 포함한 전체 파일명(STRING)
        
        === 프로그램 제어 옵션 정보 ===
        • controlOption/singleBlock - 싱글 블록 실행 여부(BOOLEAN)
        • controlOption/dryRun - 드라이 런 실행 여부(BOOLEAN)
        • controlOption/optionalStop - 옵셔널 스톱(M01) 활성화 여부(BOOLEAN)
        • controlOption/blockSkip - 블록 스킵 활성화 여부 리스트 (BOOLEAN)
        • controlOption/machineLock - 머신 락 활성화 여부(BOOLEAN)

        예시:
        - params = {"machine": 1, "channel": 1}
        - endpoint = "/machine/channel/currentProgram/sequenceNumber"

        - params = {"machine": 1, "channel": 1, "modalCode": 1}
        - endpoint = "/machine/channel/currentProgram/modal/modalCode"

        - params = {"machine": 1, "channel": 1, "workOffsetValue": 1}
        - endpoint = "/machine/channel/currentProgram/currentTotalWorkOffset/workOffsetValue"
        
        - params = {"machine": 1, "channel": 1, "blockSkip": 1}
        - endpoint = "/machine/channel/currentProgram/controlOption/blockSkip"
    """,
    "공작물 좌표계의 오프셋 정보" : 
    """
    endpoint 형식:
        • 오프셋 정보: /machine/channel/workOffset/{leaf_node}

        필수 파라미터: machine=i, channel=j, workOffset=k 와 아래 각 항목별 파라미터

        === 공작물 좌표계 오프셋 정보 ===
        • workOffsetValue - G 코드 인덱스에 대한 축별 오프셋 값 (REAL)
        • workOffsetRotation - 축별 오프셋 회전량 (SIEMENS 전용)(REAL)
        • workOffsetScalingFactor - 축별 오프셋 확장량 (SIEMENS 전용) (REAL)
        • workOffsetMirroringEnabled - 축별 미러링 활성화 여부 (SIEMENS 전용) (BOOLEAN)
        • workOffsetFine - 축별 오프셋 Fine 값 (SIEMENS 전용) (REAL)

        예시:
        # G54(workOffset=1) 좌표계의 1번째 축(workOffsetValue=1) 오프셋 값을 조회
        - params = {"machine": 1, "channel": 1, "workOffset": 1, "workOffsetValue": 1}
        - endpoint = "/machine/channel/workOffset/workOffsetValue"
    """,
    "알람 정보" : 
    """
    endpoint 형식:
        • 알람 정보: /machine/channel/alarm/{leaf_node}

        === 알람 정보 ===
        • (수정하자) - 해당 계통에서 발생한 모든 알람에 대한 Text, Category, Number, raisedTimeStamp를 리스트로 나타내는 문자열(JSON 형태)(INTEGER)
        • alarmText - 알람 상세 내용 (STRING)
        • alarmCategory - 알람 유형 (STRING)
        • alarmNumber - 알람 번호 (STRING)
        • raisedTimeStamp - 알람 발생 시각 (STRING)

        예시:
        # 1번째 발생 알람의 상세 내용을 조회
        - params = {"machine": 1, "channel": 1, "alarm": 1}
        - endpoint = "/machine/channel/alarm/alarmText"
    """,
    "사용자 변수" : 
    """
    • userVariable - 사용자 변수 값(REAL)
        
        endpoint 형식: /machine/channel/variable/{leaf_node}
    """,
    "CNC 내부 PLC 메모리 데이터" : 
    """
    endpoint 형식:
        • 메모리 정보: /machine/pic/memory/{leaf_node}

        === PLC 메모리 정보 ===
        • rbitBlock - 읽기 전용 Bit 데이터 블록 (BOOLEAN)
        • bitBlock - 읽기/쓰기 가능 Bit 데이터 블록 (BOOLEAN)
        • rbyteBlock - 읽기 전용 Byte 데이터 블록 (BYTE)
        • byteBlock - 읽기/쓰기 가능 Byte 데이터 블록 (BYTE)
        • rwordBlock - 읽기 전용 Word(2byte) 데이터 블록 (WORD)
        • wordBlock - 읽기/쓰기 가능 Word(2byte) 데이터 블록 (WORD)
        • rdwordBlock - 읽기 전용 DWord(4byte) 데이터 블록 (DWORD)
        • dwordBlock - 읽기/쓰기 가능 DWord(4byte) 데이터 블록 (DWORD)
        • rqwordBlock - 읽기 전용 QWord(8byte) 데이터 블록 (QWORD)
        • qwordBlock - 읽기/쓰기 가능 QWord(8byte) 데이터 블록 (QWORD)

        예시:
        # 100번 주소의 읽기 전용 Bit 블록 값을 조회
        - params = {"machine": 1, "rbitBlock": 100}
        - endpoint = "/machine/pic/memory/rbitBlock"

        # 200번 주소의 읽기/쓰기 Word 블록 값을 조회
        - params = {"machine": 1, "wordBlock": 200}
        - endpoint = "/machine/pic/memory/wordBlock"
    """,
    "장비 공구 영역 정보" : 
    """
    endpoint 형식:
        • 일반 정보: /machine/toolArea/{leaf_node}
        • 매거진 정보: /machine/toolArea/magazine/{leaf_node}
        • T코드 기준 공구 정보: /machine/toolArea/tools/{leaf_node}
        • T코드 기준 공구 날 정보: /machine/toolArea/tools/toolEdge/{leaf_node}
        • T코드 기준 공구 수명 정보: /machine/toolArea/tools/toolEdge/toolLife/{leaf_node}
        • 등록순 기준 공구 정보: /machine/toolArea/registerTools/{leaf_node}
        • 등록순 기준 공구 날 정보: /machine/toolArea/registerTools/toolEdge/{leaf_node}
        • 등록순 기준 공구 수명 정보: /machine/toolArea/registerTools/toolEdge/toolLife/{leaf_node}

        === 일반 공구 영역 정보 ===
        • toolAreaEnabled - 해당 공구 영역 사용 가능 여부 (BOOLEAN)
        • numberOfMagazines - 사용 가능한 매거진 개수 (INTEGER)
        • numberOfRegisteredTools - 공구 영역에 등록된 총 공구 개수 (INTEGER)
        • numberOfLoadedTools - 매거진에 탑재된 총 공구 개수 (INTEGER)
        • numberOfToolGroups - 등록된 공구 그룹의 개수 (INTEGER)
        • numberOfToolOffsets - 등록된 공구 오프셋의 개수 (INTEGER)

        === 매거진 정보 ===
        • magazine/magazineEnabled - 해당 매거진 사용 가능 여부 (BOOLEAN)
        • magazine/magazineName - 매거진 이름 (SIEMENS 전용) (STRING)
        • magazine/numberOfRealLocations - 매거진의 물리적 포트(위치) 개수 (INTEGER)
        • magazine/magazinePhysicalNumber - 매거진의 물리적 번호 (INTEGER)
        • magazine/numberOfLoadedTools - 해당 매거진에 탑재된 공구 개수 (INTEGER)

        === 공구 상세 정보 ===
        # 아래 항목들은 tools와 registerTools 경로에서 동일하게 사용됩니다. (예: /machine/toolArea/tools/toolName)
        • locationNumber - 공구가 매거진에 탑재된 위치 번호 (INTEGER)
        • toolName - 공구 이름 (STRING)
        • numberOfEdges - 공구 날의 총 개수 (INTEGER)
        • toolEnabled - 공구 영역 등록 및 매거진 탑재 여부 0: 공구 영역 미등록, 매거진 미탑재 상태, 1: 공구 영역 등록, 매거진 미탑재 상태, 2: 공구 영역 등록, 매거진 탑재 상태(INTEGER) 
        • magazineNumber - 공구가 탑재된 매거진 번호 (INTEGER)
        • sisterToolNumber - 할당된 대체 공구 번호 (INTEGER)
        • toolLifeUnit - 공구 수명 측정 단위 기준 (INTEGER)
        • toolGroupNumber - 공구가 참조된 공구 그룹 번호 리스트 (LIST[INTEGER])
        • toolUseOrderNumber - 그룹 내 공구 사용 순서 (FANUC 전용) (INTEGER)
        • toolStatus - 공구의 사용 상태 0 : Not enabled, 1 : Active tool, 2 : Enabled, 4 : Disabled, 8 : Measured, 9: 미사용 공구, 10 : 정상 수명 공구, 11 : Tool data is available (using), 12 : This tool is registered (available), 13 : This tool has expired, 14 : This tool was skipped, 16 : Prewarning limit reached , 32 : Tool being changed , 64 : Fixed location coded, 128 : Tool was in use , 256 : Tool is in the buffer magazine with transport order, 512 : Ignore disabled state of tool, 1024 : Tool must be unloaded, 2048 : Tool must be loaded, 4096 : Tool is a master tool, 8192 : Reserved, 16384 : Tool is marked for 1:1 exchange, 32768 : Tool is being used as a manual tool (INTEGER)

        === 공구 날(Edge) 상세 정보 ===
        # 아래 항목들은 .../tools/toolEdge 및 .../registerTools/toolEdge 경로에서 동일하게 사용됩니다.
        • toolType - 공구 유형 0: Not defined, 10: General-purpose tool, 11: Threading tool (Siemens에서는 540), 12: Grooving tool, 13: Round-nose tool, 14: Point nose straight tool, 15: Versatile tool, 20: Drill, 21: Counter sink tool, 22: Flat end mill, 23: Ball end mill, 24: Tap (Siemens에서는 240), 25: Reamer, 26: Boring tool, 27: Face mill, 50: Radius end mill, 51: 면취, 52: 선삭, 53: 홈삽입, 54: 나사절삭, 55: 선삭드릴, 56: 선삭탭, 100: Milling tool, 110: Ball nose end mill, 111: Conical ball end, 120: End mill, 121: End mill corner rounding, 130: Angle head cutter, 131: Corner rounding angle head cutter, 140: Facing tool, 145: Thread cutter, 150: Side mill, 151: Saw, 155: Bevelled cutter, 156: Bevelled cutter corner, 157: Tap. die-sink. cutter, 160: Drill&thread cut., 200: Twist drill, 205: Solid drill, 210: Boring bar, 220: Center drill, 230: Countersink, 231: Counterbore, 240: Tap, 241: Fine tap, 242: Tap, Whitworth, 250: Reamer, 500: Roughing tool, 510: Finishing tool, 520: Plunge cutter, 530: Cutting tool, 540: Threading tool, 550: Button tool, 560: Rotary drill, 580: 3D turning probe, 585: Calibrating tool, 700: Slotting saw, 710: 3D probe, 711: Edge finder, 712: Mono probe, 713: L probe, 714: Star probe, 725: Calibrating tool, 730: Stop, 731: Mandrel, 732: Steady rest, 900: Auxiliary tools (INTEGER) (INTEGER)
        • lengthOffsetNumber - 공구 길이 보정 식별 번호 (INTEGER)
        • toolEdge/lengthOffsetNumber - 공구 길이 보정 식별 번호 (INTEGER)
        • toolEdge/geoLengthOffset - 공구 길이 X 보정값 (REAL)
        • toolEdge/wearLengthOffset - 공구 길이 X 마모 보정값 (REAL)
        • toolEdge/radiusOffsetNumber - 공구 반경 보정 식별 번호 (INTEGER)
        • toolEdge/geoRadiusOffset - 공구 반경 보정값 (REAL)
        • toolEdge/wearRadiusOffset - 공구 반경 마모 보정값 (REAL)
        • toolEdge/edgeEnabled - 공구 날 사용 가능 여부 (BOOLEAN)
        • toolEdge/geoLengthOffsetZ - 공구 길이 Z 보정값(REAL)
        • toolEdge/wearLengthOffsetZ - 공구 길이 Z 마모 보정값(REAL)
        • toolEdge/geoLengthOffsetY - 공구 길이 Y 보정값 (REAL)
        • toolEdge/wearLengthOffsetY - 공구 길이 Y 마모 보정값 (REAL)
        • toolEdge/geoOffsetNumber - 길이 X,Z, 반경의 식별 번호 (INTEGER)
        • toolEdge/wearOffsetNumber - 길이 X,Z, 반경 마모값의 식별 번호 (INTEGER)
        • toolEdge/cuttingEdgePosition - 공구 인선 방향 (INTEGER)
        • toolEdge/tipAngle - 공구의 팁 각도 (REAL)
        • toolEdge/holderAngle - 공구 홀더 각도 (REAL)
        • toolEdge/insertAngle - 공구 인서트 각도 (REAL)
        • toolEdge/insertWidth - 인선 너비 (SIEMENS 전용) (REAL)
        • toolEdge/insertLength - 인선 길이 (SIEMENS 전용) (REAL)
        • toolEdge/referenceDirectionHolderAngle - 홀더 각도 참조 방향 (SIEMENS 전용) (REAL)
        • toolEdge/directionOfSpindleRotation - 스핀들 회전 방향 (SIEMENS 전용) (INTEGER)
        • toolEdge/numberOfTeeth - 공구 날 개수 (SIEMENS 전용) (INTEGER)

        === 공구 수명 상세 정보 ===
        # 아래 항목들은 .../toolEdge/toolLife 경로에서 동일하게 사용됩니다.
        • toolLife/maxToolLife - 최대 공구 수명 (REAL)
        • toolLife/restToolLife - 잔여 공구 수명 (REAL)
        • toolLife/toolLifeCount - 현재 공구 사용량 (REAL)
        • toolLife/toolLifeAlarm - 공구 수명 도달 경고 설정값 (REAL)

        예시:
        # 1번 공구 영역의 매거진 개수 조회
        - params = {"machine": 1, "toolArea": 1}
        - endpoint = "/machine/toolArea/numberOfMagazines"

        # T코드 5번 공구의 이름 조회
        - params = {"machine": 1, "toolArea": 1, "tools": 5}
        - endpoint = "/machine/toolArea/tools/toolName"

        # T코드 5번, 1번 날(Edge), 1번 그룹의 길이 X 보정값 조회
        - params = {"machine": 1, "toolArea": 1, "tools": 5, "toolEdge": 1, "geoLengthOffset": 1}
        - endpoint = "/machine/toolArea/tools/toolEdge/geoLengthOffset"
        
        # 등록순 3번 공구, 1번 날(Edge), 1번 그룹의 잔여 수명 조회
        - params = {"machine": 1, "toolArea": 1, "registerTools": 3, "toolEdge": 1, "restToolLife": 1}
        - endpoint = "/machine/toolArea/registerTools/toolEdge/toolLife/restToolLife"
    """,
    "내장 센서 데이터의 시계열 수집 정보" : 
    """
    endpoint 형식:
        • 버퍼 정보: /machine/buffer/{leaf_node}
        • 스트림 정보: /machine/buffer/stream/{leaf_node}

        === 버퍼 정보 ===
        • bufferEnabled - 해당 버퍼 사용 가능 여부 (BOOLEAN)
        • numberOfStream - 해당 버퍼의 최대 스트림 개수 (INTEGER)
        • statusOfStream - 스트림 상태 (0: 설정 가능, 1: 수집 가능, 2: 수집 대기, 3: 수집 중, 4: 수집 대기 혹은 수집 중, 5: 수집 완료/종료, -1: CNC 연결 실패, -2: 설정값 적용 실패 등)(필수: buffer=j) (INTEGER)
        • modOfStream - 스트림 수집 모드 (0: 반복 수집, 1: 1회 수집) (INTEGER)
        • machineChannelOfStream - 스트림 수집 시 사용할 채널(INTEGER)
        • periodOfStream - 1회 수집 기간 (단위: ms) (INTEGER)
        • triggerOfStream - 수집 시작 트리거 (0: 즉시, 1이상: 시퀀스 번호)(INTEGER)
        • frequencyOfStream - 모든 스트림에 공통으로 적용할 수집 주파수 (Hz)(INTEGER)

        === 스트림 채널 정보 ===
        • stream/streamEnabled - 해당 스트림 사용 가능 여부 (BOOLEAN)
        • stream/streamFrequency - 해당 스트림의 수집 주파수 (Hz) (INTEGER)
        • stream/streamCategory - 수집 대상 데이터 카테고리 (INTEGER)
        • stream/streamSubcategory - 수집 대상 데이터 서브카테고리 (축/스핀들 번호 등)(INTEGER)
        • stream/streamType - 수집 유형 (KCNC 전용) (INTEGER)
        • stream/streamStartBit - 수집 유형이 Bit일 때 Start Bit (KCNC 전용) (INTEGER)
        • stream/streamEndBit - 수집 유형이 Bit일 때 End Bit (KCNC 전용) (INTEGER)
        • stream/value - 해당 스트림에서 마지막으로 수집된 데이터 값(REAL)

        예시:
        # 1번 버퍼의 수집 상태를 조회
        - params = {"machine": 1, "buffer": 1}
        - endpoint = "/machine/buffer/statusOfStream"

        # 1번 버퍼의 3번 스트림에서 마지막으로 수집된 값을 조회
        - params = {"machine": 1, "buffer": 1, "stream": 3}
        - endpoint = "/machine/buffer/stream/value"
    """
}

SHORT_MANUAL = """

[
  {
    "category": "장비 상태 및 기본 정보",
    "keywords": [
      "cncModel",
      "numberOfChannels",
      "cncVendor",
      "ncLinkState",
      "currentAccessLevel",
      "basicLengthUnit",
      "machinePowerOnTime",
      "currentCncTime",
      "machineType",
      "totalCapacity",
      "usedCapacity",
      "freeCapacity",
      "rootPath"
    ],
    "description": "NC 모델명, 제조사, 통신 상태, 시간, 메모리 용량 등 장비의 정적이고 기본적인 상태 정보를 조회합니다."
  },
  {
    "category": "계통 별 기록되는 채널의 상태 정보",
    "keywords": [
      "channelEnabled",
      "toolAreaNumber",
      "numberOfAxes",
      "numberOfSpindles",
      "alarmStatus",
      "numberOfAlarms",
      "operateMode",
      "numberOfWorkOffsets",
      "ncState",
      "motionStatus",
      "emergencyStatus"
    ],
    "description": "각 채널(계통)의 활성화 여부, 알람 상태, 운전 모드(JOG/AUTO 등), NC 작동 상태 등을 조회합니다."
  },
  {
    "category": "축 별 상태 정보",
    "keywords": [
      "machinePosition",
      "workPosition",
      "distanceToGo",
      "relativePosition",
      "axisName",
      "relativeAxisName",
      "axisLoad",
      "axisFeed",
      "axisLimitPlus",
      "axisLimitMinus",
      "workAreaLimitPlus",
      "workAreaLimitMinus",
      "workAreaLimitPlusEnabled",
      "workAreaLimitMinusEnabled",
      "axisEnabled",
      "interlockEnabled",
      "constantSurfaceSpeedControlEnabled",
      "axisCurrent",
      "machineOrigin",
      "axisTemperature",
      "actualPowerConsumption",
      "powerConsumption",
      "regeneratedPower"
    ],
    "description": "각 축의 기계/공작물 좌표, 부하, 이송 속도, 리미트 설정 및 전력 소비량을 조회합니다."
  },
  {
    "category": "스핀들 별 상태 정보",
    "keywords": [
      "spindleLoad",
      "spindleOverride",
      "spindleLimit",
      "spindleEnabled",
      "spindleCurrent",
      "spindleTemperature",
      "commandedSpeed",
      "actualSpeed",
      "speedUnit",
      "actualPowerConsumption",
      "powerConsumption",
      "regeneratedPower"
    ],
    "description": "스핀들의 부하, 오버라이드, 회전 속도(RPM), 온도 및 전력 데이터를 조회합니다."
  },
  {
    "category": "축 이송 정보",
    "keywords": [
      "feedOverride",
      "rapidOverride",
      "commandedSpeed",
      "actualSpeed",
      "speedUnit"
    ],
    "description": "가공 이송 및 급속 이송의 오버라이드 비율과 지령/실제 이송 속도를 조회합니다."
  },
  {
    "category": "가공 작업의 진척 상태 정보",
    "keywords": [
      "currentWorkCounter",
      "targetWorkCounter",
      "totalWorkCounter",
      "processingMachiningTime",
      "estimatedMachiningTime",
      "machineOperationTime",
      "actualCuttingTime"
    ],
    "description": "현재/목표/총 가공 수량(Counter)과 가공 진행 시간, 실제 절삭 시간 등 생산성 관련 지표를 조회합니다."
  },
  {
    "category": "현재 활성된 공구의 상세 정보",
    "keywords": [
      "locationNumber",
      "toolName",
      "toolNumber",
      "numberOfEdges",
      "toolEnabled",
      "magazineNumber",
      "sisterToolNumber",
      "toolLifeUnit",
      "toolGroupNumber",
      "toolUseOrderNumber",
      "toolStatus",
      "edgeNumber",
      "toolType",
      "lengthOffsetNumber",
      "geoLengthOffset",
      "wearLengthOffset",
      "radiusOffsetNumber",
      "geoRadiusOffset",
      "wearRadiusOffset",
      "edgeEnabled",
      "geoLengthOffsetZ",
      "wearLengthOffsetZ",
      "geoLengthOffsetY",
      "wearLengthOffsetY",
      "geoOffsetNumber",
      "wearOffsetNumber",
      "cuttingEdgePosition",
      "tipAngle",
      "holderAngle",
      "insertAngle",
      "insertWidth",
      "insertLength",
      "referenceDirectionHolderAngle",
      "directionOfSpindleRotation",
      "numberOfTeeth",
      "maxToolLife",
      "restToolLife",
      "toolLifeCount",
      "toolLifeAlarm"
    ],
    "description": "현재 장비가 잡고 있는 활성 공구의 제원, 형상 보정값(Offset), 마모값, 수명 정보를 상세 조회합니다."
  },
  {
    "category": "현재 실행 중인 NC 프로그램의 상태 정보",
    "keywords": [
      "sequenceNumber",
      "currentBlockCounter",
      "lastBlock",
      "currentBlock",
      "nextBlock",
      "activePartProgram",
      "programMode",
      "currentWorkOffsetIndex",
      "currentWorkOffsetCode",
      "currentDepthLevel",
      "modalIndex",
      "modalCode",
      "blockCounter",
      "programName",
      "depthLevel",
      "blockData",
      "searchType",
      "mainProgramName",
      "workOffsetIndex",
      "workOffsetValue",
      "workOffsetRotation",
      "workOffsetScalingFactor",
      "workOffsetMirroringEnabled",
      "programPath",
      "programSize",
      "programDate",
      "programNameWithPath",
      "singleBlock",
      "dryRun",
      "optionalStop",
      "blockSkip",
      "machineLock"
    ],
    "description": "실행 중인 G코드 블록, 시퀀스 번호, 모달 코드, 프로그램 파일 정보 및 제어 옵션을 조회합니다."
  },
  {
    "category": "공작물 좌표계의 오프셋 정보",
    "keywords": [
      "workOffsetValue",
      "workOffsetRotation",
      "workOffsetScalingFactor",
      "workOffsetMirroringEnabled",
      "workOffsetFine"
    ],
    "description": "G54, G55 등 공작물 좌표계의 설정값(오프셋, 회전, 스케일링 등)을 조회합니다."
  },
  {
    "category": "알람 정보",
    "keywords": [
      "alarm",
      "alarmText",
      "alarmCategory",
      "alarmNumber",
      "raisedTimeStamp"
    ],
    "description": "장비에서 발생한 알람의 상세 내용, 유형, 번호, 발생 시각을 조회합니다."
  },
  {
    "category": "사용자 변수",
    "keywords": [
      "userVariable"
    ],
    "description": "NC 내부의 사용자 정의 변수 값을 조회합니다."
  },
  {
    "category": "CNC 내부 PLC 메모리 데이터",
    "keywords": [
      "rbitBlock",
      "bitBlock",
      "rbyteBlock",
      "byteBlock",
      "rwordBlock",
      "wordBlock",
      "rdwordBlock",
      "dwordBlock",
      "rqwordBlock",
      "qwordBlock"
    ],
    "description": "CNC 내부 PLC 메모리의 각 데이터 타입별(Bit, Byte, Word 등) 주소 값을 직접 조회합니다."
  },
  {
    "category": "장비 공구 영역 정보",
    "keywords": [
      "toolAreaEnabled",
      "numberOfMagazines",
      "numberOfRegisteredTools",
      "numberOfLoadedTools",
      "numberOfToolGroups",
      "numberOfToolOffsets",
      "magazineEnabled",
      "magazineName",
      "numberOfRealLocations",
      "magazinePhysicalNumber",
      "locationNumber",
      "toolName",
      "toolNumber",
      "numberOfEdges",
      "toolEnabled",
      "magazineNumber",
      "sisterToolNumber",
      "toolLifeUnit",
      "toolGroupNumber",
      "toolUseOrderNumber",
      "toolStatus",
      "toolType",
      "lengthOffsetNumber",
      "geoLengthOffset",
      "wearLengthOffset",
      "radiusOffsetNumber",
      "geoRadiusOffset",
      "wearRadiusOffset",
      "edgeEnabled",
      "geoLengthOffsetZ",
      "wearLengthOffsetZ",
      "geoLengthOffsetY",
      "wearLengthOffsetY",
      "geoOffsetNumber",
      "wearOffsetNumber",
      "cuttingEdgePosition",
      "tipAngle",
      "holderAngle",
      "insertAngle",
      "insertWidth",
      "insertLength",
      "referenceDirectionHolderAngle",
      "directionOfSpindleRotation",
      "numberOfTeeth",
      "maxToolLife",
      "restToolLife",
      "toolLifeCount",
      "toolLifeAlarm"
    ],
    "description": "전체 공구 영역, 매거진 상태 및 등록된 모든 공구의 상세 정보(제원, 오프셋, 수명)를 관리하고 조회합니다."
  },
  {
    "category": "내장 센서 데이터의 시계열 수집 정보",
    "keywords": [
      "bufferEnabled",
      "numberOfStream",
      "statusOfStream",
      "modOfStream",
      "machineChannelOfStream",
      "periodOfStream",
      "triggerOfStream",
      "frequencyOfStream",
      "streamEnabled",
      "streamFrequency",
      "streamCategory",
      "streamSubcategory",
      "streamType",
      "streamStartBit",
      "streamEndBit",
      "value"
    ],
    "description": "고속 데이터 수집을 위한 버퍼 및 스트림 설정과 수집된 시계열 센서 데이터를 조회합니다."
  }
]

"""

SHORTER_MANUAL =  """

{
    "category": "장비 상태 및 기본 정보",
    "description": "NC 컨트롤러의 모델명, 제조사, 시스템 시간 및 통신 연결 상태를 포함한 장비의 기본 제원과, NC 메모리의 전체 용량, 현재 사용량, 잔여 용량 및 파일 루트 경로 등 시스템 리소스 현황을 포괄적으로 조회합니다."
  },
  {
    "category": "계통 별 기록되는 채널의 상태 정보",
    "description": "각 채널(계통)의 활성화 여부, 현재 운전 모드(JOG, MEMORY 등)와 NC 작동 상태(Run, Stop, Hold), 비상 정지 및 알람 발생 여부를 모니터링하며, 해당 채널에 구성된 축과 스핀들의 개수 정보를 제공합니다."
  },
  {
    "category": "축 별 상태 정보",
    "description": "각 구동 축의 실시간 좌표(기계, 공작물, 상대 좌표)와 잔여 이동 거리, 축에 걸리는 부하, 전류, 온도를 모니터링하고, 소프트웨어 리미트 및 작업 금지 영역 설정 상태, 전력 소비량(실소비/회생) 데이터를 상세히 제공합니다."
  },
  {
    "category": "스핀들 별 상태 정보",
    "description": "스핀들의 지령 회전수와 실제 측정된 RPM, 속도 오버라이드 비율, 부하율 및 전류/온도 상태를 실시간으로 확인하며, 스핀들 구동에 따른 전력 소비 및 회생 전력 데이터를 조회합니다."
  },
  {
    "category": "축 이송 정보",
    "description": "가공 경로 이동(Feed) 및 급속 이송(Rapid) 시 적용된 오버라이드 비율과 단위 시간당 이송 속도의 지령값 및 실제 측정값을 비교 분석할 수 있는 데이터를 제공합니다."
  },
  {
    "category": "가공 작업의 진척 상태 정보",
    "description": "현재 진행 중인 작업의 목표 수량 대비 가공 실적(Counter)을 추적하고, 사이클 타임, 실제 절삭 시간, 자동 운전 모드 가동 시간 및 예상 잔여 시간 등 생산성 분석을 위한 시간 데이터를 제공합니다."
  },
  {
    "category": "현재 활성된 공구의 상세 정보",
    "description": "현재 가공에 사용 중이거나 스핀들에 장착된 활성 공구의 물리적 위치와 식별 정보(T-Code)를 비롯하여, 공구 날(Edge)별 형상 보정값(길이/반경), 마모 보정값, 툴 타입, 각도 제원 및 잔여 수명 정보를 상세히 조회합니다."
  },
  {
    "category": "현재 실행 중인 NC 프로그램의 상태 정보",
    "description": "실행 중인 프로그램의 파일 경로, 시퀀스 번호, 현재/이전/다음 블록 코드 내용을 추적하고, 활성화된 G-코드 모달 상태, 중단점 정보, 그리고 싱글 블록이나 드라이 런 같은 실행 제어 옵션의 활성화 여부를 제공합니다."
  },
  {
    "category": "공작물 좌표계의 오프셋 정보",
    "description": "G54, G55 등 공작물 좌표계 설정과 관련하여 각 축별 오프셋 값, 좌표 회전량, 스케일링 비율, 미러링 적용 여부 및 Fine 오프셋 등 가공 원점 보정에 필요한 데이터를 조회합니다."
  },
  {
    "category": "알람 정보",
    "description": "장비 운영 중 발생한 모든 알람에 대하여 알람 번호, 분류 카테고리, 상세 메시지 텍스트, 그리고 알람이 발생한 정확한 시각(타임스탬프) 정보를 실시간 리스트로 제공합니다."
  },
  {
    "category": "사용자 변수",
    "description": "NC 프로그램 내에서 공정 제어나 상태 저장을 위해 활용되는 사용자 정의 변수의 현재 값을 실시간으로 조회하여 로직 흐름을 모니터링합니다."
  },
  {
    "category": "CNC 내부 PLC 메모리 데이터",
    "description": "CNC 내부 PLC 메모리 영역의 특정 주소에 직접 접근하여 Bit, Byte, Word, DWord, QWord 등 다양한 데이터 타입의 값을 읽거나 모니터링할 수 있는 로우 레벨 데이터 접근 기능을 제공합니다."
  },
  {
    "category": "장비 공구 영역 정보",
    "description": "장비의 전체 매거진 구성 및 상태를 조회하고, 등록된 모든 공구의 목록과 각 공구별 상세 제원(형상/마모 보정), 수명 상태, 그룹 정보 등을 통합적으로 관리 및 조회합니다."
  },
  {
    "category": "내장 센서 데이터의 시계열 수집 정보",
    "description": "고속 데이터 분석을 위해 설정된 버퍼와 스트림의 상태를 관리하며, 밀리초(ms) 단위의 고주파수로 수집된 축/스핀들 부하, 전류 등의 시계열 센서 데이터를 제공합니다."
  }

"""