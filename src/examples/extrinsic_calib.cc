#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <iomanip>
#include <iostream>
#include <Eigen/Eigen>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <thread>
#include <limits>

#ifdef HAVE_OPENCV3
#include <opencv2/imgproc.hpp>
#else
#include <opencv2/imgproc/imgproc.hpp>
#endif // HAVE_OPENCV3

#ifdef HAVE_CUDA
#ifdef HAVE_OPENCV3
#include <opencv2/core/cuda.hpp>
#else // HAVE_OPENCV3
#include <opencv2/gpu/gpu.hpp>
namespace cv {
  namespace cuda = gpu;
}
#endif // HAVE_OPENCV3
#endif // HAVE_CUDA

#include "camodocal/calib/CamRigOdoCalibration.h"
#include "camodocal/camera_models/CameraFactory.h"

int
main(int argc, char** argv)
{
    using namespace camodocal;
    namespace fs = ::boost::filesystem;
    
    //Eigen::initParallel();

    /* 设置配置变量 */
    std::string calibDir;           // 包含标定文件路径
    std::string odoEstimateFile;    // 导航估计保存文件
    int cameraCount;                // 相机数量
    float focal;                    // 焦距
    std::string outputDir;          // 保存标定时的数据
    int nMotions;                   // 标定开始所需要的运动帧数
    int beginStage;
    bool preprocessImages;          // 是否处理图片（使用mask图像）
    bool optimizeIntrinsics;        // 是否优化内参
    std::string dataDir;            // 数据路径
    bool verbose;                   // 是否详细输出
    std::string inputDir;           // 图像输入路径 camera_%02d_%05d.png
    float refCameraGroundHeight;    // 参考相机（相机0）距离地面的高度
    float keyframeDistance;         // 关键帧距离
    std::string eventFile;          // 记录标定时产生的事件日志

    //================= Handling Program options ==================
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("calib,c", boost::program_options::value<std::string>(&calibDir)->default_value("calib"), "Directory containing camera calibration files.")
        ("estimate,e", boost::program_options::value<std::string>(&odoEstimateFile), "File containing estimate for the extrinsic calibration.")
        ("camera-count", boost::program_options::value<int>(&cameraCount)->default_value(1), "Number of cameras in rig.")
        ("f", boost::program_options::value<float>(&focal)->default_value(300.0f), "Nominal focal length.")
        ("output,o", boost::program_options::value<std::string>(&outputDir)->default_value("calibration_data"), "Directory to write calibration data to.")
        ("motions,m", boost::program_options::value<int>(&nMotions)->default_value(500), "Number of motions for calibration.")
        ("begin-stage", boost::program_options::value<int>(&beginStage)->default_value(0), "Stage to begin from.")
        ("preprocess", boost::program_options::bool_switch(&preprocessImages)->default_value(false), "Preprocess images.")
        ("optimize-intrinsics", boost::program_options::bool_switch(&optimizeIntrinsics)->default_value(false), "Optimize intrinsics in BA step.")
        ("data", boost::program_options::value<std::string>(&dataDir)->default_value("data"), "Location of folder which contains working data.")
        ("input", boost::program_options::value<std::string>(&inputDir)->default_value("input"), "Location of the folder containing all input data. Files must be named camera_%02d_%05d.png. In case if event file is specified, this is the path where to find frame_X/ subfolders")
        ("event", boost::program_options::value<std::string>(&eventFile)->default_value(std::string("")), "Event log file to be used for frame and pose events.")
        ("ref-height", boost::program_options::value<float>(&refCameraGroundHeight)->default_value(0), "Height of the reference camera (cam=0) above the ground (cameras extrinsics will be relative to the reference camera)")
        ("keydist", boost::program_options::value<float>(&keyframeDistance)->default_value(0.4), "Distance of rig to be traveled before taking a keyframe (distance is measured by means of odometry poses)")
        ("verbose,v", boost::program_options::bool_switch(&verbose)->default_value(false), "Verbose output")
        ;

    // 一些花里胡哨的配置，看不懂
    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    boost::program_options::notify(vm);

    // 打印配置参数
    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 1;
    }

    // 检查是否存在包含内参标定文件的路径
    if (!boost::filesystem::exists(calibDir))
    {
        std::cout << "# ERROR: Directory " << calibDir << " does not exist." << std::endl;
        return 1;
    }

    std::cout << "# INFO: Initializing... " << std::endl << std::flush;

    // 开始标定，默认从头开始
    if (beginStage > 0)
    {
// 是否存在CUDA
#ifdef HAVE_CUDA 
        // 检查设备是否有GPU设备
        cv::cuda::DeviceInfo info;
        if (cv::cuda::getCudaEnabledDeviceCount() > 0 && info.isCompatible())
        {
            cv::cuda::setDevice(0);     // 使用ID为0的GPU设备
            cv::cuda::resetDevice();    // 设备初始化

            cv::Mat dummy(1, 1, CV_8UC1);
            dummy = cv::Scalar(0);

            cv::cuda::GpuMat dummyGPU;
            dummyGPU.upload(dummy);     // 测试GPU能够正常加载图像

            dummyGPU.release();         // 释放资源
        }
        else
        {
            std::cout << "# ERROR: No Cuda device found!\n";
            exit(1);
        }
#else  // HAVE_CUDA
        std::cout << "# ERROR: Application not compiled with CUDA! Either recompile with CUDA or modify this program to work without it.\n";
        exit(1);
#endif // HAVE_CUDA
    }

    //========================= Handling Input =======================

    //===========================Initialize calibration==========================

    // 读取相机参数
    std::vector<camodocal::CameraPtr> cameras(cameraCount);
    for (int i = 0; i < cameraCount; ++i)
    {
        // step1: 创建相机对象，这里使用了工厂模式和单例模式结合的方法构建相机对象。
        camodocal::CameraPtr camera;
        {
            boost::filesystem::path calibFilePath(calibDir);

            std::ostringstream oss;
            oss << "camera_" << i << "_calib.yaml";
            calibFilePath /= oss.str();

            camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(calibFilePath.string());
            if (camera.get() == 0)
            {
                std::cout << "# ERROR: Unable to read calibration file: " << calibFilePath.string() << std::endl;
                return 0;
            }
        }

        // step2：读取每个相机的mask图像
        {
            boost::filesystem::path maskFilePath(calibDir);

            std::ostringstream oss;
            oss << "camera_" << i << "_mask.png";
            maskFilePath /= oss.str();

            cv::Mat mask = cv::imread(maskFilePath.string());
            if (!mask.empty())
            {
                cv::Mat grey;
                cv::cvtColor(mask, grey, CV_RGB2GRAY, 1);
                camera->mask() = grey;
                std::cout << "# INFO: Foudn camera mask for camera " << camera->cameraName() << std::endl;
            }
        }


        cameras.at(i) = camera;
    }

    // 读取估计的外参。个人理解为如果不是重头开始计算，则需要读取外参；否则不需要读取外参；
    /*  外参格式
        r11 r12 r13 tx
        r21 r22 r23 ty
        r31 r32 r33 tz
        0   0   0   1
    */
    std::map<unsigned, Eigen::Matrix4d, std::less<unsigned>, Eigen::aligned_allocator<std::pair<const unsigned, Eigen::Matrix4d> > > estimates;
    if (odoEstimateFile.length())   // 如果存在文件
    {
        std::cout << "# INFO: parse extrinsic calibration estimates file " << odoEstimateFile << std::endl;

        std::ifstream file(odoEstimateFile);    // 加载文件
        if (file.is_open())  // 打开文件
        {
            std::string line;
            while(getline(file, line))  // 读取行
            {
                auto it = std::find_if(cameras.begin(), cameras.end(), [&line](camodocal::CameraPtr cam)
                { return cam && boost::iequals(cam->cameraName(), line); });

                if (it == cameras.end()) continue;

                std::cout << "# INFO: found estimate for camera " << line << std::endl;

                Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
                file >> T(0,0) >> T(0,1) >> T(0,2) >> T(0,3);
                file >> T(1,0) >> T(1,1) >> T(1,2) >> T(1,3);
                file >> T(2,0) >> T(2,1) >> T(2,2) >> T(2,3);

                estimates[std::distance(cameras.begin(), it)] = T;
            }
        }
    }


    //========================= Get all files  =========================
    typedef std::map<int64_t, std::string>  ImageMap;
    typedef std::map<int64_t, Eigen::Isometry3f, std::less<int64_t>, Eigen::aligned_allocator<std::pair<const int64_t, Eigen::Isometry3f> > > IsometryMap;

    std::vector< ImageMap > inputImages(cameraCount);   // 输入的图像数据，<timestamped, imagepath>
    IsometryMap inputOdometry;      // 输入的导航数据，<timestamped, transformation>
    bool bUseGPS = false;           // 是否使用GPS
    if (eventFile.length() == 0)    // 如果不存在配置路径
    {
        printf("Get images and pose files out from result directory\n");

        fs::path inputFilePath(inputDir);   // 输入的图像

        fs::recursive_directory_iterator it(inputFilePath);
        fs::recursive_directory_iterator endit;

        while (it != endit)
        {
            if (fs::is_regular_file(*it) && it->path().extension() == ".png")
            {
                int camera = -1;
                uint64_t timestamp = 0;

                if (sscanf(it->path().filename().c_str(), "camera_%d_%lu.png", &camera, &timestamp) != 2)
                {
                    printf("cannot find input image camera_[d]_[llu].png\n");
                    return 1;
                }
                printf("image name : %s time : %ld", it->path().string().c_str(), timestamp);
                inputImages[camera][timestamp] = it->path().string();
            }

            if (fs::is_regular_file(*it) && it->path().extension() == ".txt" && it->path().filename().string().find_first_of("pose_") == 0)
            {
                uint64_t timestamp = 0;
                if (sscanf(it->path().filename().c_str(), "pose_%lu.txt", &timestamp) != 1)
                {
                    printf("pose filename %s has a wrong name, must be pose_[llu].txt\n", it->path().filename().c_str());
                    return 1;
                }

                // read pose
                Eigen::Vector3f t;
                Eigen::Matrix3f R;
                std::ifstream file(it->path().c_str());
                std::cout << "pose path : " << it->path().c_str() << std::endl;
                if (!file.is_open())
                {
                    printf("cannot find file %s containg a valid pose\n", it->path().c_str());
                    return 1;
                }

                file >> R(0,0) >> R(0, 1) >> R(0, 2);
                file >> R(1,0) >> R(1, 1) >> R(1, 2);
                file >> R(2,0) >> R(2, 1) >> R(2, 2);
                file >> t[0] >> t[1] >> t[2];
                
                file.close();


                Eigen::Isometry3f T;

                T.matrix().block<3,3>(0,0) = R;
                T.matrix().block<3,1>(0,3) = t;
                inputOdometry[timestamp] = T;
            }

            it++;
        }
    }else   // 如果存在配置路径
    {
        printf("Read %s file to get all the events\n", eventFile.c_str());

        std::ifstream file(eventFile.c_str());
        if (!file.is_open())
        {
            printf("Cannot open %s", eventFile.c_str());
            return 1;
        }

        // read line by line and interpret accordin event
        std::string line;
        Eigen::Quaternionf lastIMU(0,0,0,1);
        while(std::getline(file, line))
        {
            std::stringstream str(line);

            // type of event
            unsigned long long timestamp = 0;
            std::string type;

            str >> timestamp >> type;

            if (type.compare("CAM") == 0)
            {
                int camid = 0;
                std::string frame;
                str >> camid >> frame;
                inputImages[camid][timestamp] = inputDir + "/frames_" + boost::lexical_cast<std::string>(camid) + "/" + frame;
                //printf("image [%d][%llu] = %s\n", camid, timestamp, inputImages[camid][timestamp].c_str());
            }else if (type.compare("IMU") == 0)
            {
                str >> lastIMU.x() >> lastIMU.y() >> lastIMU.z() >> lastIMU.w();
            }else if (type.compare("GPS") == 0)
            {
                Eigen::Vector3f gps(0,0,0);
                str >> gps[0] >> gps[1] >> gps[2];

                // construct the odometry entry
                Eigen::Isometry3f T;
                T.matrix().block<3,3>(0,0) = lastIMU.toRotationMatrix();
                T.matrix().block<3,1>(0,3) = gps;
                inputOdometry[timestamp] = T;

                bUseGPS = true;
            }
        }
    }

    //========================= Start Threads =========================


    /* 只有特征点均匀的分布整个图像才会优化内参 */
    CamRigOdoCalibration::Options options;
//    options.mode = CamRigOdoCalibration::ONLINE;
    options.poseSource = bUseGPS ? PoseSource::GPS_INS : PoseSource::ODOMETRY;
    options.nMotions = nMotions;
    options.minKeyframeDistance = keyframeDistance;
    options.minVOSegmentSize = 15;
    options.preprocessImages = preprocessImages;
    options.optimizeIntrinsics = optimizeIntrinsics;
    options.saveWorkingData = true;
    options.beginStage = beginStage;
    options.dataDir = dataDir;
    options.verbose = verbose;

    // 创建标定对象
    CamRigOdoCalibration camRigOdoCalib(cameras, options);

    // 设置外参初始估计参数
    for(auto it : estimates) camRigOdoCalib.setInitialCameraOdoTransformEstimates(it.first, it.second);

    std::cout << "# INFO: Initialization finished!" << std::endl;

    // 标定线程
    // 输入： 图像，里程计，相机对象，相机数量，是否使用GPS
    std::thread inputThread([&inputImages, &inputOdometry, &camRigOdoCalib, cameraCount, bUseGPS]()
    {
        //uint64_t lastTimestamp = std::numeric_limits<uint64_t>::max();

        std::vector<ImageMap::iterator> camIterator(cameraCount);   // 图像迭代器
        IsometryMap::iterator locIterator = inputOdometry.begin();  // 里程计迭代器
        
        // 拷贝图像信息
        for (int c=0; c < cameraCount; c++)
            camIterator[c] = inputImages[c].begin();

        // 拷贝里程计信息
        auto addLocation = [&camRigOdoCalib, bUseGPS](uint64_t timestamp, const Eigen::Isometry3f& T)
        {
            if (bUseGPS)    // 使用GPS
            {
                Eigen::Quaternionf q(T.rotation());     // gps旋转信息
                Eigen::Vector3f gps = T.translation();  // gps位移信息
                camRigOdoCalib.addGpsIns(gps[0], gps[1], gps[2], q.x(), q.y(), q.z(), q.w(), timestamp); // 添加GPS信息

                // 打印信息
                std::cout << "GPS: lat=" << gps[0] << ", lon=" << gps[1] << ", alt=" << gps[2]
                          << ", qx=" << q.x() << ", qy=" << q.y() << ", qz=" << q.z() << ", qw=" << q.w()
                          << " [" << timestamp << "]" << std::endl;
            }else   // 不使用GPS
            {
                float yaw = std::atan2(T.linear()(1,0), T.linear()(0,0));   // 计算yaw角
                camRigOdoCalib.addOdometry(T.translation()[0], T.translation()[1], T.translation()[2], yaw, timestamp); // 添加里程计信息

                // 打印信息
                std::cout << "POSE: x=" << T.translation()[0] << ", y=" << T.translation()[1] << ", yaw=" << yaw << " [" << timestamp << "]" << std::endl;
            }
        };

        // 添加定位信息，使用的是上面的addLocation函数
        for (int i=0; i < 3 && locIterator != inputOdometry.end(); i++, locIterator++)
        {
            addLocation(locIterator->first, locIterator->second);
        }

        // 输入的里程计信息不为空
        while(locIterator != inputOdometry.end())
        {
            if (camRigOdoCalib.isRunning()) break;  // 如果当前相机线程正在运行，则退出；

            int64_t locTime = locIterator->first;   // 时间戳
            addLocation(locTime, locIterator->second);  // 添加定位信息

            // 添加图像信息
            bool hasData = true;    // flag设为真
            while(hasData)
            {
                hasData = false;    // flag设为假
                for (int c=0; c < cameraCount; c++)
                {
                    if(camIterator[c] == inputImages[c].end()) continue;    // 如果图像为空，则继续；
                    // 图像在里程信息之前，这里是不是应该以图像时间戳对其呢？
                    // 而且判断条件过于简单了
                    if(camIterator[c]->first < locTime)  
                    {
                        uint64_t camTime = camIterator[c]->first;   // 相机时间戳
                        std::cout << "IMG: " << camTime << " -> " << camIterator[c]->second << std::endl;   // 打印图像信息
                        std::cout << "Pose : " << locIterator->first << std::endl << locIterator->second.linear() << std::endl; // 打印里程计信息
                        camRigOdoCalib.addFrame(c, cv::imread(camIterator[c]->second), camTime);    // 添加图像信息
                        camIterator[c]++;   // 相机c读取下一帧
                        hasData = true;     // // flag设为真
                    }
                }
            }

            locIterator++;  // 读取下一帧里程计信息
        }

#if 0
        //int ignore_frame = 3;

        //std::ofstream pose_dump("pose_dump.obj");
        //int last_vertex_idx = 1;

        //for (size_t i=0; i < inputOdometry.size() && !camRigOdoCalib.isRunning(); i++)
        for (const auto& pair : inputOdometry)
        {
            if (camRigOdoCalib.isRunning()) break;

            uint64_t timestamp = pair.first;
            const Eigen::Isometry3f& T = pair.second;

            // dump oriented box
            /*{
                std::vector<Eigen::Vector3f> vertex(4);
                std::vector<Eigen::Vector3f> color(4);
                vertex[0] = Eigen::Vector3f(2,1,0);  color[0] = Eigen::Vector3f(255,0,0);
                vertex[1] = Eigen::Vector3f(2,-1,0); color[1] = Eigen::Vector3f(0,255,0);
                vertex[2] = Eigen::Vector3f(-2,1,0); color[2] = Eigen::Vector3f(0,0,255);
                vertex[3] = Eigen::Vector3f(-2,-1,0);color[3] = Eigen::Vector3f(255,0,255);

                for(Eigen::Vector3f v : vertex)
                {
                    v = T * v;
                    pose_dump << "v " << v[0] << " " << v[1] << " " << v[2] << " "  << color[0] << " " << color[1] << " " << color[2] << std::endl;
                    last_vertex_idx++;
                }
                pose_dump << "f " << last_vertex_idx-4 << " " << last_vertex_idx-3 << " " << last_vertex_idx-2 << std::endl;
                pose_dump << "f " << last_vertex_idx-2 << " " << last_vertex_idx-3 << " " << last_vertex_idx-1 << std::endl;
            }*/

            // frames (make sure that sensor data is always fresher than the image data)
            for (int c=0; c < cameraCount && timestamp > lastTimestamp; c++)
            {
                if (inputImages[c].find(lastTimestamp) != inputImages[c].end())
                {
                    std::cout << "read " << inputImages[c][lastTimestamp] << std::endl << std::flush;
                    //frames[c] = cv::imread(inputImages[c][lastTimestamp]);
                    camRigOdoCalib.addFrame(c, cv::imread(inputImages[c][lastTimestamp]), lastTimestamp);
                }
            }

            if (ignore_frame-- < 0)
                lastTimestamp = timestamp;
        }
#endif

        if (!camRigOdoCalib.isRunning()) camRigOdoCalib.run();  // 运行标定程序
    });


    //****************
    //
    // IMPORTANT: Create a thread, and in this thread,
    //            add data in the order of increasing timestamp
    //            with one important exception for offline mode:
    //            ensure that before you add a frame with timestamp t,
    //            you have already added either odometry or GPS/INS data
    //            with a timestamp greater than t, depending on the
    //            pose source you are calibrating against.
    //
    // Add odometry and image data here.
    // camRigOdoCalib.addOdometry(x, y, yaw, timestamp);
    // camRigOdoCalib.addFrame(cameraId, image, timestamp);
    //
    // Alternatively, if you are calibrating against GPS/INS,
    // set options.poseSource = GPS_INS, and add GPS/INS
    // and image data here.
    //
    // camRigOdoCalib.addGpsIns(lat, lon, alt, roll, pitch, yaw, timestamp);
    // camRigOdoCalib.addFrame(cameraId, image, timestamp);
    //
    // If options.mode == CamRigOdoCalibration::ONLINE,
    // the addFrame call returns immediately.
    // If options.mode == CamRigOdoCalibration::OFFLINE,
    // the addFrame call returns after the image has been processed.
    //
    // After you are done, if the minimum number of motions has not been
    // reached, but you want to run the calibration anyway, call:
    // camRigOdoCalib.run();
    //
    //****************

    // Receive and process incoming data. Calibration automatically runs
    // once minimum number of motions has been reached for all cameras.
    // Check camRigOdoCalib.running() to see if the calibration is running.
    // If so, you can stop adding data. To run the calibration without
    // waiting for the minimum motion requirement to be met,
    // camRigOdoCalib.run();
    camRigOdoCalib.start();
    
    CameraSystem cameraSystem = camRigOdoCalib.cameraSystem();
    cameraSystem.setReferenceCamera(0);
    cameraSystem.writeToDirectory(outputDir);

    std::cout << "# INFO: Wrote calibration data to " << outputDir << "." << std::endl;

    std::cout << std::fixed << std::setprecision(5);

    /*std::cout << "# INFO: Current estimate (local):" << std::endl;
    for (int i = 0; i < cameraCount; ++i)
    {
        const Eigen::Matrix4d& H = cameraSystem.getLocalCameraPose(i);
        std::cout << "========== Camera " << i << " ==========" << std::endl;
        std::cout << "Rotation: " << std::endl;
        std::cout << H.block<3,3>(0,0) << std::endl;

        std::cout << "Translation: " << std::endl;
        std::cout << H.block<3,1>(0,3).transpose() << std::endl;
    }*/


    float camHeightDiff = cameraSystem.getGlobalCameraPose(0)(2,3) - refCameraGroundHeight;
    std::cout << "# INFO: Current estimate (global):" << std::endl;
    for (int i = 0; i < cameraCount; ++i)
    {
        Eigen::Matrix4d H = cameraSystem.getGlobalCameraPose(i);
        //H.block<3,1>(0,1) *= -1;
        //H.block<3,1>(0,2) *= -1;
        Eigen::Quaterniond Q(H.block<3,3>(0,0));
        Eigen::Vector3d T = H.block<3,1>(0,3);

        T[2] -= camHeightDiff;

        std::cout << "========== Camera " << i << " ==========" << std::endl;
        std::cout << "Rotation: " << std::endl;
        std::cout << H.block<3,3>(0,0) << std::endl;

        std::cout << "Rotation Q: " << std::endl;
        std::cout << " " << Q.x() << " " << Q.y() << " " << Q.z() << " " << Q.w() << std::endl;

        std::cout << "Translation: " << std::endl;
        std::cout << T.transpose() << std::endl << std::endl;
    }
    inputThread.join();

    return 0;
}
