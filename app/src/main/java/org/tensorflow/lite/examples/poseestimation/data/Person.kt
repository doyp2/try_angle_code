package org.tensorflow.lite.examples.poseestimation.data

import android.graphics.RectF
import android.graphics.PointF


data class Person(
    var id: Int = -1, // default id is -1
    val keyPoints: List<KeyPoint>,
    val boundingBox: RectF? = null, // Only MoveNet MultiPose return bounding box.
    val score: Float
){
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
        val necessaryParts = setOf(BodyPart.LEFT_ANKLE, BodyPart.RIGHT_ANKLE, BodyPart.NOSE)
        val detectedParts = keyPoints.map { it.bodyPart }.toSet()
        return necessaryParts.all { it in detectedParts }
    }
}


