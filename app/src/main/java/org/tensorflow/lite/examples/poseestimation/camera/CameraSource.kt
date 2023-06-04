package org.tensorflow.lite.examples.poseestimation.camera

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.*
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.media.Image
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.util.Size
import android.view.Surface
import android.view.SurfaceView
import kotlinx.coroutines.Runnable
import kotlinx.coroutines.suspendCancellableCoroutine
import org.tensorflow.lite.examples.poseestimation.MainActivity
import org.tensorflow.lite.examples.poseestimation.VisualizationUtils
import org.tensorflow.lite.examples.poseestimation.YuvToRgbConverter
import org.tensorflow.lite.examples.poseestimation.data.Person
import org.tensorflow.lite.examples.poseestimation.ml.MoveNetMultiPose
import org.tensorflow.lite.examples.poseestimation.ml.PoseClassifier
import org.tensorflow.lite.examples.poseestimation.ml.PoseDetector
import org.tensorflow.lite.examples.poseestimation.ml.TrackerType
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.net.Socket
import java.util.*
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException


class CameraSource(
    private val surfaceView: SurfaceView,
    private val listener: CameraSourceListener? = null,
) {
    var latestPerson: Person? = null

    // Socket과 DataOutputStream을 멤버 변수로 추가
    private var socket: Socket? = null
    private var socket_motor: Socket? = null
    private var dataOutputStream: DataOutputStream? = null
    private var dataInputStream: DataInputStream? = null
    private var dataOutputStream_motor: DataOutputStream? = null

    private var thred_runnig:Boolean = false
    private var check_person:Boolean = false
    private var jpegBytes:ByteArray? = null

    var StreamingThread:HandlerThread? = null
    var MotorThread:HandlerThread? = null
    var StreamingHandler:Handler? = null
    var MotorHandler:Handler? = null

    // 라즈베리 파이로 데이터 전송
    private fun sendToRaspberryPi(message: String) {
        try {
            // 소켓과 DataOutputStream이 이미 열려 있는지 확인
            if (socket_motor == null || socket_motor!!.isClosed) {
                return
            }
            dataOutputStream_motor?.write(message.toByteArray())
            dataOutputStream_motor?.flush()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun sendToImg(img : ByteArray) {
        try {
            // 소켓과 DataOutputStream이 이미 열려 있는지 확인
            if (socket == null || socket!!.isClosed) {
                return
            }

            dataInputStream?.readByte() // start 수신
            val imageSizeBytes = img.size.toString().toByteArray()
            dataOutputStream?.write(imageSizeBytes)

            dataInputStream?.readByte() // image 수신
            dataOutputStream?.write(img)
            dataOutputStream?.flush()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    // 앱이 시작될 때 Socket과 DataOutputStream을 초기화
    private fun start() {
        try {
            socket = Socket("192.168.2.1", 9000)
            dataOutputStream = DataOutputStream(socket!!.getOutputStream())
            dataInputStream = DataInputStream(socket!!.getInputStream())
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
    private fun start_motor() {
        try {
            socket_motor = Socket("192.168.2.1", 8888)
            dataOutputStream_motor = DataOutputStream(socket_motor!!.getOutputStream())
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    // 앱이 종료될 때 Socket과 DataOutputStream을 닫음
    private fun stop() {
        Thread {
            try {
                dataOutputStream?.close()
                dataInputStream?.close()
                socket?.close()

                dataOutputStream_motor?.close()
                socket_motor?.close()

//                motor_handler_stop()
//                stream_handler_stop()
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }.start()
    }
    // 카메라 이미지를 JPG 바이트로 변환하는 함수
    private fun convertImageToJpeg(bitmap: Bitmap): ByteArray {
        val outputStream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, outputStream)
        return outputStream.toByteArray()
    }

    private fun checkCameraSupport(): Boolean { // 카메라 지원확인 코드
        val characteristics = cameraManager.getCameraCharacteristics(cameraId)
        val map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)

        // Suppose you need to check support for ImageFormat.YUV_420_888 with a size of 1920x1080
        val outputSizes = map?.getOutputSizes(ImageFormat.YUV_420_888)
        outputSizes?.let { sizes ->
            return sizes.contains(Size(1920, 1080)) // 해상도 설정
        }

        // If the outputSizes is null or the desired size isn't supported, return false
        return false
    }

    companion object { // 정적 코드
        private const val PREVIEW_WIDTH = 640
        private const val PREVIEW_HEIGHT = 480
        private const val PREVIEW_WIDTH_T = 480
        private const val PREVIEW_HEIGHT_T = 640

        /** Threshold for confidence score. */
        private const val MIN_CONFIDENCE = 0.3f
        private const val TAG = "Camera Source"
    }

    private val lock = Any()
    private var detector: PoseDetector? = null
    private var classifier: PoseClassifier? = null
    private var isTrackerEnabled = false
    private var yuvConverter: YuvToRgbConverter = YuvToRgbConverter(surfaceView.context)
    private lateinit var imageBitmap: Bitmap

    /** Frame count that have been processed so far in an one second interval to calculate FPS. */
    private var fpsTimer: Timer? = null
    private var frameProcessedInOneSecondInterval = 0
    private var framesPerSecond = 0

    /** Detects, characterizes, and connects to a CameraDevice (used for all camera operations) */
    private val cameraManager: CameraManager by lazy {
        val context = surfaceView.context
        context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    }

    /** Readers used as buffers for camera still shots */
    private var imageReader: ImageReader? = null

    /** The [CameraDevice] that will be opened in this fragment */
    private var camera: CameraDevice? = null

    /** Internal reference to the ongoing [CameraCaptureSession] configured with our parameters */
    private var session: CameraCaptureSession? = null

    /** [HandlerThread] where all buffer reading operations run */
    private var imageReaderThread: HandlerThread? = null

    /** [Handler] corresponding to [imageReaderThread] */
    private var imageReaderHandler: Handler? = null
    private var cameraId: String = ""

    suspend fun initCamera() {
        checkCameraSupport()
        thred_runnig = true
        StreamingThread = HandlerThread("StreamingThread")
        MotorThread = HandlerThread("MotorThread")
        StreamingThread?.start()
        MotorThread?.start()
        StreamingHandler = StreamingThread?.looper?.let { Handler(it) }
        MotorHandler = MotorThread?.looper?.let { Handler(it) }
        StreamingHandler?.post{
            start()
            while (thred_runnig){
                jpegBytes?.let { sendToImg(it) }
                SystemClock.sleep(250)
            }
        }
        MotorHandler?.post{
            start_motor()
            while (thred_runnig){
                if (check_person){
                    sendToRaspberryPi("@@@")
                }
                SystemClock.sleep(500)
            }
        }
        camera = openCamera(cameraManager, cameraId)
        imageReader = ImageReader.newInstance(PREVIEW_WIDTH_T, PREVIEW_HEIGHT_T, ImageFormat.YUV_420_888, 3)
        imageReader?.setOnImageAvailableListener({ reader -> // ImageReader가 새로운 이미지를 사용가능할 때 호출될 리스너 등록
            val image = reader.acquireLatestImage() // 큐에서 최신 이미지를 가져오고 오래된 이미지 삭제
            if (image != null) {
                if (!::imageBitmap.isInitialized) {
                    imageBitmap =
                        Bitmap.createBitmap(
                            PREVIEW_WIDTH,
                            PREVIEW_HEIGHT,
                            Bitmap.Config.ARGB_8888
                        )
                }
                yuvConverter.yuvToRgb(image, imageBitmap)
                // Create rotated version for portrait display
                jpegBytes = convertImageToJpeg(imageBitmap)
                val rotateMatrix = Matrix()
                rotateMatrix.postRotate(90.0f)

                val rotatedBitmap = Bitmap.createBitmap(
                    imageBitmap, 0, 0, PREVIEW_WIDTH, PREVIEW_HEIGHT,
                    rotateMatrix, false
                )
                processImage(rotatedBitmap)
                image.close()
            }
        }, imageReaderHandler)

        imageReader?.surface?.let { surface ->
            session = createSession(listOf(surface))
            val cameraRequest = camera?.createCaptureRequest(
                CameraDevice.TEMPLATE_PREVIEW
            )?.apply {
                addTarget(surface)
            }
            cameraRequest?.build()?.let {
                session?.setRepeatingRequest(it, null, null)
            }
        }
    }

    private suspend fun createSession(targets: List<Surface>): CameraCaptureSession =
        suspendCancellableCoroutine { cont ->
            camera?.createCaptureSession(targets, object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(captureSession: CameraCaptureSession) =
                    cont.resume(captureSession)

                override fun onConfigureFailed(session: CameraCaptureSession) {
                    cont.resumeWithException(Exception("Session error"))
                }
            }, null)
        }

    @SuppressLint("MissingPermission")
    private suspend fun openCamera(manager: CameraManager, cameraId: String): CameraDevice =
        suspendCancellableCoroutine { cont ->
            manager.openCamera(cameraId, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) = cont.resume(camera)

                override fun onDisconnected(camera: CameraDevice) {
                    camera.close()
                }

                override fun onError(camera: CameraDevice, error: Int) {
                    if (cont.isActive) cont.resumeWithException(Exception("Camera error"))
                }
            }, imageReaderHandler)
        }

    fun prepareCamera() { // 어떤 카메라를 쓸지 결정하는 함수
        for (cameraId in cameraManager.cameraIdList) {
            val characteristics = cameraManager.getCameraCharacteristics(cameraId)
            val deviceLevel = characteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)
            if (deviceLevel == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL){ // camera api를 다 지원하는 카메라로 설정
                this.cameraId = cameraId
            }
        }
    }

    fun setDetector(detector: PoseDetector) {
        synchronized(lock) {
            if (this.detector != null) {
                this.detector?.close()
                this.detector = null
            }
            this.detector = detector
        }
    }

    fun setClassifier(classifier: PoseClassifier?) {
        synchronized(lock) {
            if (this.classifier != null) {
                this.classifier?.close()
                this.classifier = null
            }
            this.classifier = classifier
        }
    }


    /**
     * Set Tracker for Movenet MuiltiPose model.
     */
    fun setTracker(trackerType: TrackerType) {
        isTrackerEnabled = trackerType != TrackerType.OFF
        (this.detector as? MoveNetMultiPose)?.setTracker(trackerType)
    }

    fun resume() {
        imageReaderThread = HandlerThread("imageReaderThread").apply { start() }
        imageReaderHandler = Handler(imageReaderThread!!.looper)
        fpsTimer = Timer()
        fpsTimer?.scheduleAtFixedRate(
            object : TimerTask() {
                override fun run() {
                    framesPerSecond = frameProcessedInOneSecondInterval
                    frameProcessedInOneSecondInterval = 0
                }
            },
            0,
            1000
        )
    }

    fun close() {
        stop()
        thred_runnig = false
        check_person = false
        session?.close()
        session = null
        camera?.close()
        camera = null
        imageReader?.close()
        imageReader = null
        stopImageReaderThread()
        detector?.close()
        detector = null
        classifier?.close()
        classifier = null
        fpsTimer?.cancel()
        fpsTimer = null
        frameProcessedInOneSecondInterval = 0
        framesPerSecond = 0
    }

    // process image 이미지 분석 코드 ******************
    private fun processImage(bitmap: Bitmap) {
        var setDir = 0
        var check_center : Boolean = false
        val persons = mutableListOf<Person>()
        var classificationResult: List<Pair<String, Float>>? = null
        synchronized(lock) {
            detector?.estimatePoses(bitmap)?.let {
                persons.addAll(it)
                // if the model only returns one item, allow running the Pose classifier.
                if (persons.isNotEmpty()) {
                    classifier?.run {
                        classificationResult = classify(persons[0])
                    }
                }
            }
        }
        frameProcessedInOneSecondInterval++
        if (frameProcessedInOneSecondInterval == 1) {
            // send fps to view
            listener?.onFPSListener(framesPerSecond)
        }
        // if the model returns only one item, show that item's score.
        if (persons.isNotEmpty()) {
            listener?.onDetectedInfo(persons[0].score, classificationResult)
        }
        // 사람 객체가 있으면 실행
        if (persons.isNotEmpty()) {
            check_person = true
            val person = persons[0]
            if (MainActivity.adjustMode) { // 전신확인 단계일 경우 ***************
                if(person.score > MIN_CONFIDENCE) {
                    val (ankle, shoulder) = person.getAnkleAndShoulder()

                    if (ankle != null) {
                        val targetAnkleY = bitmap.height.toFloat() - 60f
                        if (ankle.y < targetAnkleY) {
                            showToast("발목이 하단으로부터 ${targetAnkleY - ankle.y}만큼 위에 있습니다. 아래로 이동해주세요.")
                        } else {
                            showToast("발목이 하단으로부터 ${ankle.y - targetAnkleY}만큼 아래에 있습니다. 위로 이동해주세요.")
                        }
                    }
                }
//                if (shoulder != null) {
//                    val targetShoulderY = bitmap.height * 1 / 3f
//                    if (shoulder.y < targetShoulderY) {
//                        showToast("어깨가 상단 2/3 지점으로부터 ${targetShoulderY - shoulder.y}만큼 위에 있습니다. 아래로 이동해주세요.")
//                    } else {
//                        showToast("어깨가 상단 2/3 지점으로부터 ${shoulder.y - targetShoulderY}만큼 아래에 있습니다. 위로 이동해주세요.")
//                    }
//                }
            } else { // 중앙확인 단계일 경우 **************
                val center = person.getCenter()
                if (center != null) {
                    val targetX = bitmap.width / 2f // 중앙 x 좌표
                    val targetY = bitmap.height / 2f // 중앙 y 좌표
                    val distanceX = targetX - center.x // 중앙 x 좌표 - 사람 x 좌표
                    val distanceY = targetY - center.y // 중앙 y 좌표 - 사람 y 좌표

//                    if(person.score > MIN_CONFIDENCE){
//                        showToast("객체가 중앙으로부터 X: ${distanceX}, Y: ${distanceY}만큼 떨어져 있습니다.")
//                    }
                    // 중앙 확인 코드
                    val widthThird = bitmap.width / 3f // 480/3 = 120
                    val heightThird = bitmap.height / 3f // 640/3 = 213.33
                    val isCenterXInMiddleGrid = center.x > widthThird + 40f && center.x < (2*widthThird) - 40f
                    val isCenterYInMiddleGrid = center.y > heightThird + 80f && center.y < (2*heightThird) - 80f
                    val personBoundingBox = person.boundingBox
                    if (personBoundingBox != null) {
                        val isPersonInFrame = personBoundingBox.left >= 0 && personBoundingBox.top >= 0 &&
                                personBoundingBox.right <= PREVIEW_WIDTH_T && personBoundingBox.bottom <= PREVIEW_HEIGHT_T
                        if (person.isFullBodyDetected() && isPersonInFrame) {
                            // The person is in the center grid and entire person is within the frame
                            if (isCenterXInMiddleGrid && isCenterYInMiddleGrid) {
                                check_center = true
                                showToast("가운데에 있다")
                            } else { // 가운데에 있지않다면
                                // The object is out of the center grid, show the distance to the center grid
                                val distanceToCenterGridX = when {
                                    center.x <= widthThird -> widthThird - center.x
                                    else -> center.x - 2 * widthThird
                                }
                                val distanceToCenterGridY = when {
                                    center.y <= heightThird -> heightThird - center.y
                                    else -> center.y - 2 * heightThird
                                }
                                setDir = if (center.x < targetX - 10f){
                                    -1
                                } else if (center.x > targetX + 10f){
                                    1
                                } else{
                                    if (center.y < targetY - 10f){
                                        -2
                                    }
                                    else if (center.y > targetY + 10f){
                                        2
                                    }
                                    else{
                                        0
                                    }
                                }
//                                showToast("Out of center grid by \n dx=$distanceToCenterGridX, dy=$distanceToCenterGridY")
                            }
                        }
                    }
                }
            }
        }
        else{
            check_person = false
        }
        visualize(persons, bitmap, check_center, setDir)
    }
    // In CameraSource class
    //    showToast 메소드가 메인 스레드에서 호출되면 즉시 onDistanceUpdate 메소드를 호출하고,
    //    그렇지 않으면 Handler를 사용하여 메인 스레드에서 실행되도록 예약합니다. 이렇게 하면 딜레이를 최소화할 수 있습니다.
    private fun showToast(message: String) {
        if (Looper.myLooper() == Looper.getMainLooper()) {
            listener?.onDistanceUpdate(message)
        } else {
            Handler(Looper.getMainLooper()).post {
                listener?.onDistanceUpdate(message)
            }
        }
    }

    private fun visualize(persons: List<Person>, bitmap: Bitmap, check_center: Boolean, setDir: Int) {
        // +check_center 변수 추가해서 중앙이면 다르게 그리게
        val setC : Int = if (check_center) { Color.GREEN } else { Color.RED }
        val outputBitmap = VisualizationUtils.drawBodyKeypoints(
            bitmap,
            persons.filter { it.score > MIN_CONFIDENCE },
            isTrackerEnabled,
            setC,
            setDir
        )

        val holder = surfaceView.holder
        val surfaceCanvas = holder.lockCanvas()
        surfaceCanvas?.let { canvas ->
            val screenWidth: Int
            val screenHeight: Int
            val left: Int
            val top: Int

            if (canvas.height > canvas.width) {
                val ratio = outputBitmap.height.toFloat() / outputBitmap.width
                screenWidth = canvas.width
                left = 0
                screenHeight = (canvas.width * ratio).toInt()
                top = (canvas.height - screenHeight) / 2
            } else {
                val ratio = outputBitmap.width.toFloat() / outputBitmap.height
                screenHeight = canvas.height
                top = 0
                screenWidth = (canvas.height * ratio).toInt()
                left = (canvas.width - screenWidth) / 2
            }
            val right: Int = left + screenWidth
            val bottom: Int = top + screenHeight

            canvas.drawBitmap(
                outputBitmap, Rect(0, 0, outputBitmap.width, outputBitmap.height),
                Rect(left, top, right, bottom), null
            )
            surfaceView.holder.unlockCanvasAndPost(canvas)
        }
    }

    private fun stopImageReaderThread() {
        imageReaderThread?.quitSafely()
        try {
            imageReaderThread?.join()
            imageReaderThread = null
            imageReaderHandler = null
        } catch (e: InterruptedException) {
            Log.d(TAG, e.message.toString())
        }
    }

    interface CameraSourceListener {
        fun onFPSListener(fps: Int)

        fun onDetectedInfo(personScore: Float?, poseLabels: List<Pair<String, Float>>?)

        fun onDistanceUpdate(message: String)
    }
}
