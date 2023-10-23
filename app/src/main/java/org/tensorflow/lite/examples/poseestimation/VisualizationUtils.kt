/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================
*/

package org.tensorflow.lite.examples.poseestimation

import android.graphics.*
import org.tensorflow.lite.examples.poseestimation.data.BodyPart
import org.tensorflow.lite.examples.poseestimation.data.Person
import kotlin.math.max
import org.tensorflow.lite.examples.poseestimation.camera.CameraSource

object VisualizationUtils {
    /** Radius of circle used to draw keypoints.  */
    private const val CIRCLE_RADIUS = 6f

    /** Width of line used to connected two keypoints.  */
    private const val LINE_WIDTH = 4f

    /** The text size of the person id that will be displayed when the tracker is available.  */
    private const val PERSON_ID_TEXT_SIZE = 30f

    /** Distance from person id to the nose keypoint.  */
    private const val PERSON_ID_MARGIN = 6f

    /** Pair of keypoints to draw lines between.  */
    private val bodyJoints = listOf(
        Pair(BodyPart.NOSE, BodyPart.LEFT_EYE),
        Pair(BodyPart.NOSE, BodyPart.RIGHT_EYE),
        Pair(BodyPart.LEFT_EYE, BodyPart.LEFT_EAR),
        Pair(BodyPart.RIGHT_EYE, BodyPart.RIGHT_EAR),
        Pair(BodyPart.NOSE, BodyPart.LEFT_SHOULDER),
        Pair(BodyPart.NOSE, BodyPart.RIGHT_SHOULDER),
        Pair(BodyPart.LEFT_SHOULDER, BodyPart.LEFT_ELBOW),
        Pair(BodyPart.LEFT_ELBOW, BodyPart.LEFT_WRIST),
        Pair(BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_ELBOW),
        Pair(BodyPart.RIGHT_ELBOW, BodyPart.RIGHT_WRIST),
        Pair(BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER),
        Pair(BodyPart.LEFT_SHOULDER, BodyPart.LEFT_HIP),
        Pair(BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_HIP),
        Pair(BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP),
        Pair(BodyPart.LEFT_HIP, BodyPart.LEFT_KNEE),
        Pair(BodyPart.LEFT_KNEE, BodyPart.LEFT_ANKLE),
        Pair(BodyPart.RIGHT_HIP, BodyPart.RIGHT_KNEE),
        Pair(BodyPart.RIGHT_KNEE, BodyPart.RIGHT_ANKLE)
    )

    // Draw line and point indicate body pose
    fun drawBodyKeypoints(
        input: Bitmap,
        person: Person?,
        isTrackerEnabled: Boolean = false,
        setColor: Int,
        dir: String
    ): Bitmap {
        val paintCircle = Paint().apply { // 관절 원
            strokeWidth = CIRCLE_RADIUS
            color = setColor
            style = Paint.Style.FILL
        }
        val paintCircleCenter = Paint().apply { // 중앙 점
            strokeWidth = CIRCLE_RADIUS
            color = Color.BLUE
            style = Paint.Style.FILL
        }
        val paintLine = Paint().apply { // 관절 라인
            strokeWidth = LINE_WIDTH
            color = setColor
            style = Paint.Style.STROKE
        }

        val paintText = Paint().apply {
            textSize = PERSON_ID_TEXT_SIZE
            color = Color.BLUE
            textAlign = Paint.Align.LEFT
        }

        val output = input.copy(Bitmap.Config.ARGB_8888, true)
        val originalSizeCanvas = Canvas(output)

        // Add this part to draw 3x3 grid
        val gridPaint = Paint().apply {
            color = Color.WHITE
            strokeWidth = 2f
        }

        val widthThird = input.width / 3f
        val heightThird = input.height / 3f

        for (i in 1..2) {
            originalSizeCanvas.drawLine(
                i * widthThird,
                0f,
                i * widthThird,
                input.height.toFloat(),
                gridPaint
            )
            originalSizeCanvas.drawLine(
                0f,
                i * heightThird,
                input.width.toFloat(),
                i * heightThird,
                gridPaint
            )
        }
        if (person != null) {
            // draw person id if tracker is enable
            if (isTrackerEnabled) {
                person.boundingBox?.let {
                    val personIdX = max(0f, it.left)
                    val personIdY = max(0f, it.top)

                    originalSizeCanvas.drawText(
                        person.id.toString(),
                        personIdX,
                        personIdY - PERSON_ID_MARGIN,
                        paintText
                    )
                    originalSizeCanvas.drawRect(it, paintLine)
                }
            }
            bodyJoints.forEach {
                val pointA = person.keyPoints[it.first.position].coordinate
                val pointB = person.keyPoints[it.second.position].coordinate
                originalSizeCanvas.drawLine(pointA.x, pointA.y, pointB.x, pointB.y, paintLine)
            }

            person.keyPoints.forEach { point ->
                originalSizeCanvas.drawCircle(
                    point.coordinate.x,
                    point.coordinate.y,
                    CIRCLE_RADIUS,
                    paintCircle
                )
            }

            // 중심부 점 찍기
            fun drawCenter(): PointF? {
                val centerPoints =
                    person.keyPoints.filter { it.bodyPart == BodyPart.NOSE || it.bodyPart == BodyPart.LEFT_HIP || it.bodyPart == BodyPart.RIGHT_HIP }
                // Check if centerPoints is empty before calculating average x, y coordinates
                if (centerPoints.isEmpty()) {
                    return null
                }
                val avgX = centerPoints.map { it.coordinate.x }.average().toFloat()
                val avgY = centerPoints.map { it.coordinate.y }.average().toFloat()

                originalSizeCanvas.drawCircle(
                    avgX,
                    avgY,
                    CIRCLE_RADIUS,
                    paintCircleCenter
                )
                return PointF(avgX, avgY)
            }
            drawCenter()

            val paintArrow = Paint().apply {
                strokeWidth = LINE_WIDTH
                color = Color.RED
                style = Paint.Style.STROKE
            }

            if (dir == "left") {
                originalSizeCanvas.drawLine(40f, 320f, 120f, 320f, paintArrow)
                originalSizeCanvas.drawLine(40f, 320f, 60f, 340f, paintArrow)
                originalSizeCanvas.drawLine(40f, 320f, 60f, 300f, paintArrow)
            }
            if (dir == "right") {
                originalSizeCanvas.drawLine(360f, 320f, 440f, 320f, paintArrow)
                originalSizeCanvas.drawLine(420f, 340f, 440f, 320f, paintArrow)
                originalSizeCanvas.drawLine(420f, 300f, 440f, 320f, paintArrow)
            }
            if (dir == "up") {
                originalSizeCanvas.drawLine(240f, 20f, 240f, 60f, paintArrow)
                originalSizeCanvas.drawLine(230f, 30f, 240f, 20f, paintArrow)
                originalSizeCanvas.drawLine(250f, 30f, 240f, 20f, paintArrow)
            }
            if (dir == "down") {
                originalSizeCanvas.drawLine(240f, 620f, 240f, 580f, paintArrow)
                originalSizeCanvas.drawLine(230f, 610f, 240f, 620f, paintArrow)
                originalSizeCanvas.drawLine(250f, 610f, 240f, 620f, paintArrow)
            }
            if (dir == "front") {
                originalSizeCanvas.drawLine(240f, 20f, 240f, 60f, paintArrow)
                originalSizeCanvas.drawLine(230f, 30f, 240f, 20f, paintArrow)
                originalSizeCanvas.drawLine(250f, 30f, 240f, 20f, paintArrow)
            }
            if (dir == "back") {
                originalSizeCanvas.drawLine(240f, 620f, 240f, 580f, paintArrow)
                originalSizeCanvas.drawLine(230f, 610f, 240f, 620f, paintArrow)
                originalSizeCanvas.drawLine(250f, 610f, 240f, 620f, paintArrow)
            }
        }
        return output
    }
}
