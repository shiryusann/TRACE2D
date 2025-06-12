from mmdemo2d.demo import Demo

from mmdemo2d.features import (
    create_input_features,
    DisplayFrame,
    ObjectiveDetection,
    Bodypoints
)

if __name__ == "__main__":
    VIDEO_PATH = f"D:/multimodality/laptop_test.mp4"
    OBJ_MODEL_PATH = f"D:/multimodality/yolo11n.pt"
    BODY_MODEL_PATH = f"D:/multimodality/yolo11n-pose.pt"
    FRAME_RATE = 10
    THRESHOLD = 0.3
    PLAY_BACK = True

    colorimage, frametime = create_input_features(
        VIDEO_PATH,
        FRAME_RATE,
        PLAY_BACK
    )

    objects = ObjectiveDetection(
        OBJ_MODEL_PATH,
        THRESHOLD,
        colorimage
    )

    participants = Bodypoints(
        BODY_MODEL_PATH,
        colorimage
    )

    demo = Demo(
        targets = [
            DisplayFrame(colorimage, objects, participants)
        ]
    )

    #demo.show_dependency_graph()
    demo.run()