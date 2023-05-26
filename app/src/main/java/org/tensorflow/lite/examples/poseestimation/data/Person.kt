package org.tensorflow.lite.examples.poseestimation.data

import android.graphics.RectF
import android.graphics.PointF
import kotlin.math.hypot


data class Person(
    var id: Int = -1, // default id is -1
    val keyPoints: List<KeyPoint>,
    val boundingBox: RectF? = null, // Only MoveNet MultiPose return bounding box.
    val score: Float
){
    companion object {
        private const val MIN_CONFIDENCE = .2f
    }
    fun getCenter(): PointF? {
        val centerPoints = keyPoints.filter { it.bodyPart == BodyPart.NOSE || it.bodyPart == BodyPart.LEFT_HIP || it.bodyPart == BodyPart.RIGHT_HIP }
        // Check if centerPoints is empty before calculating average x, y coordinates
        if (centerPoints.isEmpty()) {
            return null
        }
        val avgX = centerPoints.map { it.coordinate.x }.average().toFloat()
        val avgY = centerPoints.map { it.coordinate.y }.average().toFloat()
        return PointF(avgX, avgY)
    }

    fun isFullBodyDetected(): Boolean {
        val leftKnee = keyPoints.find { it.bodyPart == BodyPart.LEFT_KNEE }
        val rightKnee = keyPoints.find { it.bodyPart == BodyPart.RIGHT_KNEE }
        val leftAnkle = keyPoints.find { it.bodyPart == BodyPart.LEFT_ANKLE }
        val rightAnkle = keyPoints.find { it.bodyPart == BodyPart.RIGHT_ANKLE }

        // Check if all required keypoints are detected
        if (leftKnee == null || rightKnee == null || leftAnkle == null || rightAnkle == null) {
            return false
        }

        val minDistance = 50 // Set this to the minimum distance you want between the knee and ankle

        // Check if both knees and ankles are detected with confidence
        if (leftKnee.score > MIN_CONFIDENCE && rightKnee.score > MIN_CONFIDENCE &&
            leftAnkle.score > MIN_CONFIDENCE && rightAnkle.score > MIN_CONFIDENCE) {

            // Check if the distance between left knee and ankle is greater than the minimum distance
            if (distanceBetweenPoints(leftKnee.coordinate, leftAnkle.coordinate) < minDistance) {
                return false
            }

            // Check if the distance between right knee and ankle is greater than the minimum distance
            if (distanceBetweenPoints(rightKnee.coordinate, rightAnkle.coordinate) < minDistance) {
                return false
            }

            // Check if the ankles are below the knees
            if (leftAnkle.coordinate.y < leftKnee.coordinate.y || rightAnkle.coordinate.y < rightKnee.coordinate.y) {
                return false
            }

            return true
        }

        return false
    }


    private fun distanceBetweenPoints(point1: PointF, point2: PointF): Float {
        return hypot((point2.x - point1.x).toDouble(), (point2.y - point1.y).toDouble()).toFloat()
    }
}


