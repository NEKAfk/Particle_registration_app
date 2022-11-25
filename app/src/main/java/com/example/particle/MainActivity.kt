package com.example.particle

import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.*
import android.graphics.Color.rgb
import android.os.Bundle
import android.util.Log
import android.view.WindowManager
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.core.Camera
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.particle.databinding.ActivityMainBinding
import org.tensorflow.lite.Interpreter
import java.io.*
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.roundToInt

class MainActivity : AppCompatActivity() {
    private lateinit var cam: Camera
    private var isWork = false
    private lateinit var binding: ActivityMainBinding
    private lateinit var imageCapture:ImageCapture
    private lateinit var imageAnalysis:ImageAnalysis
    private lateinit var outputDirectory: File
    private var tflite: Interpreter? = null
    private val names = arrayOf("Noise", "Spot", "Track", "Worm")
    private val coef = 4.237F / 1000

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        outputDirectory = getOutputDirectory()
        Toast.makeText(this@MainActivity, "$outputDirectory", Toast.LENGTH_SHORT)
            .show()

        checkPermissions(Constants.REQUIRED_PERMISSIONS, Constants.REQUEST_CODE_PERMISSION)
        if(allPermissionsGranted()) {
                startCamera()
        }
        try{
            tflite = Interpreter(loadModel())
        }
        catch(e: Exception){
            e.printStackTrace()
        }
        binding.btnTurnOn.setOnClickListener{turnOn()}
    }

    private fun getOutputDirectory(): File{
        val mediaDir = externalMediaDirs.firstOrNull()?.let { mFile->
            File(mFile, resources.getString(R.string.app_name)).apply { mkdirs() }
        }

        return if (mediaDir != null && mediaDir.exists())
            mediaDir else filesDir
    }

    @SuppressLint("SetTextI18n")
    private fun turnOn(){
        if(isWork)
        {
            isWork = false
            binding.btnTurnOn.text = "START"
        }
        else
        {
            isWork = true
            binding.btnTurnOn.text = "STOP"
        }
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun startCamera(){

        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({

            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder()
                .build()
                .also {mPreview->
                    mPreview.setSurfaceProvider(
                        binding.viewFinder.surfaceProvider
                    )
                }
            imageCapture = ImageCapture.Builder()
                .build()
            imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
            imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(this)){ image->
                //Если кнопка START нажата, то обрабатываем изображение
                if(isWork)
                {
                    //Создание массива из изображения
                    val bitmap = image.toBitmap()
                    val x = bitmap.width
                    val y = bitmap.height
                    val arr = IntArray(x*y) {0}
                    bitmap.getPixels(arr, 0, x, 0, 0, x, y)

                    //Поиск максимального значения яркости пикселя на изображении по одному из каналов
                    var mi = 0
                    var maxVal = 0
                    for(i in arr.indices)
                    {
                        val imR = Color.red(arr[i])
                        val imG = Color.green(arr[i])
                        val imB = Color.blue(arr[i])
                        if(maxVal < maxOf(imR, imG, imB)){
                            maxVal = maxOf(imR, imB, imG)
                            mi = i
                        }
                    }

                    //Если максимальное значение как мминимум 20, то обрабатываем изображение дальше
                    if(maxVal >= 20){
                        //Создание битмапа для сохранения в него вырезанного изображения
                        val bmp = Bitmap.createBitmap(50, 50, Bitmap.Config.ARGB_8888, false)
                        //Общая сумма яркостей пикселей по одному из каналов для измерения энергии частицы
                        var slum = 0F
                        //Входные данные для CNN
                        val input = Array(1){Array(50){Array(50){FloatArray(3){0f} } }}
                        for(i in -25..24){
                            for(j in -25..24){
                                //Проверяем вхождение в границы исходного изображения,
                                //так как частица могла попасть на границу
                                if((mi/x + i) >= 0 && (mi/x + i) < y && (mi%x + j) >= 0 && (mi%x + j) < x) {
                                    //считывание яркостей по трем каналам
                                    var r = Color.red(arr[mi+i*x+j]).toFloat()
                                    var g = Color.green(arr[mi+i*x+j]).toFloat()
                                    var b = Color.blue(arr[mi+i*x+j]).toFloat()
                                    //суммарная яркость пикеля
                                    val lum = (r+g+b)
                                    slum += lum/3
                                    //изменение яркости пикселя для градиента
                                    if(lum < 45){
                                        r= 1F - 1F / 45F * (45F - lum)
                                        g = 0F
                                        b = 0F
                                    }
                                    else if(lum < 90){
                                        r = 1F
                                        g = 1F - 1F / 45F * (90F - lum)
                                        b = 0F
                                    }
                                    else{
                                        r = 1F
                                        g = 1F
                                        b = 1 - 1F / 30F * (120F - lum)
                                    }

                                    input[0][j+25][i+25][0] = r*255
                                    input[0][j+25][i+25][1] = g*255
                                    input[0][j+25][i+25][2] = b*255
                                    bmp.setPixel(j+25, i+25, rgb(r, g, b))
                                }
                            }
                        }
                        //сохранение изображения в папку,
                        // которая зависит от выходного значения нейронной сети
                        saveIm(useModel(input), bmp, coef*slum)
                    }
                }
                //закрываем изображение
                //для получения на обработку нового
                image.close()
            }
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try{
                cameraProvider.unbindAll()
                 cam = cameraProvider.bindToLifecycle(
                    this, cameraSelector,
                    preview, imageCapture, imageAnalysis
                )
            }catch (e:Exception){
                Log.d(Constants.TAG, "startCamera Fail: ", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun saveIm(output: Array<FloatArray>, bmp: Bitmap, E: Float) {
        val roundE = (E*1000).roundToInt().toFloat() / 1000F
        val fileName = SimpleDateFormat(Constants.FILE_NAME_FORMAT,
            Locale.getDefault())
            .format(System.currentTimeMillis())+"_("+roundE.toString()+" MeV).png"
        var ind = 0

        for(i in 0..3){
            if(output[0][i] > output[0][ind]){
                ind = i
            }
        }

        val file: File?
        val dir: File?
        dir = File(outputDirectory.absolutePath + File.separator + names[ind])
        if(!dir.exists()){
            dir.mkdirs()
        }
        file = File(outputDirectory.absolutePath + File.separator + names[ind] + File.separator + fileName)
        file.createNewFile()

        val fos = BufferedOutputStream(FileOutputStream(file))
        bmp.compress(Bitmap.CompressFormat.PNG, 90, fos) // YOU can also save it in JPEG

        fos.close()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        for(permission in permissions) {
            if (ContextCompat.checkSelfPermission(
                    this@MainActivity,
                    permission
                ) == PackageManager.PERMISSION_DENIED
            ) {
                Toast.makeText(this@MainActivity, "$permission already denied", Toast.LENGTH_SHORT)
                    .show()
            } else {
                Toast.makeText(this@MainActivity, "$permission already granted", Toast.LENGTH_SHORT)
                    .show()
            }
        }
    }

    private fun checkPermissions(permissions: Array<String>, requestCode: Int) {
        if(!allPermissionsGranted()){
            // Requesting the permission
            ActivityCompat.requestPermissions(
                this@MainActivity,
                permissions,
                requestCode)
        }

    }

    private fun allPermissionsGranted() =
        Constants.REQUIRED_PERMISSIONS.all{
            ContextCompat.checkSelfPermission(
                this@MainActivity, it
            ) == PackageManager.PERMISSION_GRANTED
        }

    private fun ImageProxy.toBitmap(): Bitmap {
        val yBuffer = planes[0].buffer // Y
        val uBuffer = planes[1].buffer // U
        val vBuffer = planes[2].buffer // V

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, this.width, this.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 100, out)
        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    @Throws(IOException::class)
    private fun loadModel(): MappedByteBuffer{
        val fileDescriptor = this.assets.openFd("model.tflite")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val start = fileDescriptor.startOffset
        val end = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, start, end)
    }

    private fun useModel(input: Array<Array<Array<FloatArray>>>): Array<FloatArray>{
        val output = Array(1){FloatArray(4){0f}}
        tflite!!.run(input, output)
        Log.d("CHECK", "${output[0][0]} ${output[0][1]} ${output[0][2]} ${output[0][3]}")
        return output
    }
}

