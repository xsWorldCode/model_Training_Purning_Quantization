import cv2
import sys

def test_camera(index=0):
    print(f"--- 正在尝试调用摄像头 索引:{index} ---")
    
    # 在 Windows 上，cv2.CAP_DSHOW 通常能解决 'index out of range' 或 开启缓慢的问题
    # 如果你是 Linux/Mac，可以去掉 cv2.CAP_DSHOW
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print(f"❌ 索引 {index}: 无法打开。")
        return False

    # 尝试读取一帧
    ret, frame = cap.read()
    if not ret:
        print(f"❌ 索引 {index}: 成功打开但无法读取图像流（可能被占用）。")
        cap.release()
        return False

    print(f"✅ 索引 {index}: 成功打开！分辨率: {int(cap.get(3))}x{int(cap.get(4))}")
    print("🎬 正在打开实时窗口，按下 'Q' 键退出...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 在画面上显示文字提示
        cv2.putText(frame, f"Camera {index} - Press 'Q' to Exit", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(f'Camera Test Index {index}', frame)
        
        # 等待按键，Q 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"--- 索引 {index} 测试结束 ---")
    return True

if __name__ == "__main__":
    # 依次尝试索引 0, 1, 2
    for i in range(3):
        success = test_camera(i)
        if success:
            print(f"\n💡 建议：在 YOLOv5 中请使用 --source {i}")
            break
    else:
        print("\n所有索引均失败。请检查：")
        print("1. 摄像头是否已插好？")
        print("2. 隐私设置中是否允许应用访问摄像头？")
        print("3. 是否有其他软件（如微信、OBS）正在占用摄像头？")