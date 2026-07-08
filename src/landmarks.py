import math


def midpoint(p1, p2):
    return (
        (p1[0] + p2[0]) / 2.0,
        (p1[1] + p2[1]) / 2.0,
    )


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_eye_distance(landmarks):
    left_eye = landmarks["left_eye"]
    right_eye = landmarks["right_eye"]

    eye_distance = distance(left_eye, right_eye)

    if eye_distance <= 0:
        raise ValueError("Invalid landmarks: eye distance is zero.")

    return eye_distance


def get_pose_proxy(landmarks):
    """
    Very rough pose/landmark quality proxy.

    This is not real head-pose estimation.

    It tells us whether the 5 CelebA landmarks look usable for simple
    landmark-relative sticker placement.

    Useful because profile images can have both eyes very close together,
    making cheek/forehead estimates unreliable.
    """

    left_eye = landmarks["left_eye"]
    right_eye = landmarks["right_eye"]
    nose = landmarks["nose"]
    left_mouth = landmarks["left_mouth"]
    right_mouth = landmarks["right_mouth"]

    eye_distance = get_eye_distance(landmarks)
    mouth_width = distance(left_mouth, right_mouth)

    eye_center = midpoint(left_eye, right_eye)
    mouth_center = midpoint(left_mouth, right_mouth)

    # How far the nose is from the center of the eyes, normalized by eye distance.
    nose_offset_x = abs(nose[0] - eye_center[0]) / eye_distance

    # How far the mouth center is from the nose horizontally.
    mouth_offset_x = abs(mouth_center[0] - nose[0]) / eye_distance

    return {
        "eye_distance": eye_distance,
        "mouth_width": mouth_width,
        "nose_offset_x": nose_offset_x,
        "mouth_offset_x": mouth_offset_x,
    }


def landmarks_are_reasonable_for_simple_positions(landmarks):
    """
    Returns True if the landmarks are reasonable enough for the first prototype.

    This should not be used as a final scientific filter yet.
    It is only meant to avoid obviously bad geometry in early experiments.
    """

    pose = get_pose_proxy(landmarks)

    if pose["eye_distance"] < 15:
        return False

    if pose["mouth_width"] < 10:
        return False

    # In profile images, the nose can be far to one side of the eye center.
    if pose["nose_offset_x"] > 1.25:
        return False

    return True


def get_candidate_positions(landmarks, include_unreliable=False):
    """
    Creates named sticker positions from CelebA's 5 landmarks.

    Reliable positions:
        - between_eyes
        - nose
        - mouth_center
        - eye_region
        - lower_face

    Optional unreliable positions:
        - forehead
        - left_cheek
        - right_cheek

    The unreliable positions are useful for visual experiments, but should
    not be part of the first serious evaluation.
    """

    left_eye = landmarks["left_eye"]
    right_eye = landmarks["right_eye"]
    nose = landmarks["nose"]
    left_mouth = landmarks["left_mouth"]
    right_mouth = landmarks["right_mouth"]

    eye_center = midpoint(left_eye, right_eye)
    mouth_center = midpoint(left_mouth, right_mouth)
    eye_distance = get_eye_distance(landmarks)

    positions = {
        "between_eyes": eye_center,
        "nose": nose,
        "mouth_center": mouth_center,
        # Same coordinate as between_eyes, but semantically useful for
        # sunglasses/censor bars.
        "eye_region": eye_center,
        # Slightly below mouth center.
        "lower_face": (
            mouth_center[0],
            mouth_center[1] + 0.35 * eye_distance,
        ),
    }

    if include_unreliable:
        positions.update(
            {
                "forehead": (
                    eye_center[0],
                    eye_center[1] - 0.85 * eye_distance,
                ),
                "left_cheek": (
                    nose[0] - 0.95 * eye_distance,
                    nose[1] + 0.35 * eye_distance,
                ),
                "right_cheek": (
                    nose[0] + 0.95 * eye_distance,
                    nose[1] + 0.35 * eye_distance,
                ),
            }
        )

    return positions


def get_default_sticker_width(landmarks, scale_factor=1.2):
    """
    Returns a sticker width in pixels.

    scale_factor=1.0 means roughly one inter-eye distance wide.
    """

    eye_distance = get_eye_distance(landmarks)
    return eye_distance * scale_factor
