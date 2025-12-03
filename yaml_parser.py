import yaml
import os
import glob


class RPPGYamlParser:
    def __init__(self):
        # 데이터를 저장할 딕셔너리 초기화
        # 구조: {'DATASET_NAME': ['FORMAT1', 'FORMAT2']}
        self.dataset_map = {}

    def parse_directory(self, dir_path):
        """
        지정된 디렉토리 내의 모든 .yaml 파일을 찾아 파싱합니다.
        """
        # 디렉토리 존재 여부 확인
        if not os.path.exists(dir_path):
            print(f"[에러] 디렉토리를 찾을 수 없습니다: {dir_path}")
            return

        # .yaml 확장자를 가진 모든 파일 경로 검색
        search_pattern = os.path.join(dir_path, "*.yaml")
        yaml_files = glob.glob(search_pattern)

        print(f"📂 '{dir_path}'에서 {len(yaml_files)}개의 YAML 파일을 발견했습니다.")
        print("-" * 50)

        # 각 파일에 대해 파싱 수행
        for file_path in yaml_files:
            self._parse_file(file_path)

    def _parse_file(self, file_path):
        """
        개별 YAML 파일을 읽고 데이터를 추출합니다.
        """
        file_name = os.path.basename(file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # YAML 구조 탐색 (TEST -> DATA)
            # .get()을 사용하여 키가 없을 경우 에러 대신 None이나 빈 dict 반환
            test_config = config.get('TEST', {})
            if not test_config:
                print(f"[Skip] '{file_name}': 'TEST' 섹션 없음")
                return

            data_config = test_config.get('DATA', {})
            model_config = config.get('MODEL', {})
            # 타겟 데이터 추출
            model_key = model_config.get('NAME', {})
            dataset_key = data_config.get('DATASET')
            data_format_value = data_config.get('DATA_FORMAT')

            # 유효성 검사 및 저장
            if model_key and data_format_value:
                self._update_storage(model_key, data_format_value)
                print(f"[성공] '{file_name}' 처리됨: {model_key} -> {data_format_value}")
            else:
                print(f"[Skip] '{file_name}': 필요한 키(DATASET, DATA_FORMAT) 누락")

        except Exception as e:
            print(f"[에러] '{file_name}' 파싱 중 오류: {e}")

    def _update_storage(self, key, value):
        """
        딕셔너리에 데이터를 추가합니다 (중복 방지 로직 포함).
        """
        if key in self.dataset_map:
            # 이미 존재하는 데이터셋인 경우, 포맷 리스트에 없으면 추가
            if value not in self.dataset_map[key]:
                self.dataset_map[key].append(value)
        else:
            # 새로운 데이터셋인 경우 리스트 초기화
            self.dataset_map[key] = [value]

    def get_result(self):
        """최종 결과 딕셔너리를 반환합니다."""
        return self.dataset_map


# ==========================================
# 메인 실행부
# ==========================================
if __name__ == "__main__":
    # 1. 파서 인스턴스 생성
    parser = RPPGYamlParser()

    # 2. 타겟 디렉토리 설정 (요청하신 경로)
    target_directory = "configs/infer_configs"

    # 3. 디렉토리 파싱 실행
    print(f"🚀 파싱 시작: {target_directory}")
    parser.parse_directory(target_directory)

    # 4. 최종 결과 출력
    print("\n📊 최종 결과 (Dictionary):")
    print(parser.get_result())