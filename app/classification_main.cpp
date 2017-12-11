// 使用visual studio 编译运行基于tensorflow的 inceptionV3 分类识别网络 

#include <fstream>
#include <utility>
#include <vector>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include "time.h"
#include "math.h"

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <filesystem>

using namespace cv;
using namespace std;
namespace fs = std::experimental::filesystem;

#define INPUT_WIDTH  (224) //(299)
#define INPUT_HEIGHT (224) // (299)

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
// 读取标签文件，对应训练时设置的标签类别，方便检查网络正向预测的类别正确与否
Status ReadLabelsFile(const string& file_name, std::vector<string>* result,
	size_t* found_label_count) {
	std::ifstream file(file_name);
	if (!file) {
		return tensorflow::errors::NotFound("Labels file ", file_name,
			" not found.");
	}
	result->clear();
	string line;
	while (std::getline(file, line)) {
		result->push_back(line);
	}
	*found_label_count = result->size();
	const int padding = 16;
	while (result->size() % padding) {
		result->emplace_back();
	}
	return Status::OK();
}

// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
//根据文件名读取图片并转化为Tensor
Status ReadTensorFromImageFile(const string& file_name, const int input_height,
	const int input_width, const float input_mean,
	const float input_std,
	std::vector<Tensor>* out_tensors) {
	auto root = tensorflow::Scope::NewRootScope();
	using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

	string input_name = "file_reader";
	string output_name = "normalized";
	auto file_reader =
		tensorflow::ops::ReadFile(root.WithOpName(input_name), file_name);
	// Now try to figure out what kind of file it is and decode it.
	const int wanted_channels = 1;// 3; // hack this value for image channel
	tensorflow::Output image_reader;
	if (tensorflow::StringPiece(file_name).ends_with(".png")) {
		image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
			DecodePng::Channels(wanted_channels));
	}
	else if (tensorflow::StringPiece(file_name).ends_with(".gif")) {
		// gif decoder returns 4-D tensor, remove the first dim
		image_reader = Squeeze(root.WithOpName("squeeze_first_dim"),
			DecodeGif(root.WithOpName("gif_reader"),
				file_reader));
	}
	else {
		// Assume if it's neither a PNG nor a GIF then it must be a JPEG.
		image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
			DecodeJpeg::Channels(wanted_channels));
	}
	// Now cast the image data to float so we can do normal math on it.
	auto float_caster =
		Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);
	// The convention for image ops in TensorFlow is that all images are expected
	// to be in batches, so that they're four-dimensional arrays with indices of
	// [batch, height, width, channel]. Because we only have a single image, we
	// have to add a batch dimension of 1 to the start with ExpandDims().
	auto dims_expander = ExpandDims(root, float_caster, 0);
	
	// Bilinearly resize the image to fit the required dimensions.
	auto resized = ResizeBilinear(
		root, dims_expander,
		Const(root.WithOpName("size"), { input_height, input_width }));
	// Subtract the mean and divide by the scale.
	Div(root.WithOpName(output_name), Sub(root, resized, { input_mean }),
	{ input_std });

	// This runs the GraphDef network definition that we've just constructed, and
	// returns the results in the output tensor.
	tensorflow::GraphDef graph;
	TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

	std::unique_ptr<tensorflow::Session> session(
		tensorflow::NewSession(tensorflow::SessionOptions()));
	TF_RETURN_IF_ERROR(session->Create(graph));
	TF_RETURN_IF_ERROR(session->Run({}, { output_name }, {}, out_tensors));
	return Status::OK();
}

// convert opencv load image data into tensor
Status ReadOpencvfile(const string& file_name, const int input_height,
	const int input_width, const float input_mean,
	const float input_std,
	std::vector<Tensor>* out_tensors) {
	auto root = tensorflow::Scope::NewRootScope();
	using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

	string input_name = "file_reader";
	string output_name = "normalized";

	tensorflow::Tensor image_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, INPUT_HEIGHT, INPUT_WIDTH, 3}));
	auto input_tensor_mapped = image_tensor.tensor<float, 4>();

	const float* source_data = NULL;

	for (int y = 0; y < INPUT_HEIGHT; ++y) {
		const float* source_row = source_data + (y * INPUT_WIDTH * 3);
		for (int x = 0; x < INPUT_WIDTH; ++x) {
			const float* source_pixel = source_row + (x * 3);
			for (int c = 0; c < 3; ++c) {
				const float* source_value = source_pixel + c;
				input_tensor_mapped(0, y, x, c) = *source_value;
			}
		}
	}

	auto file_reader =
		tensorflow::ops::ReadFile(root.WithOpName(input_name), file_name);
	// Now try to figure out what kind of file it is and decode it.
	const int wanted_channels = 3;
	tensorflow::Output image_reader;
	if (tensorflow::StringPiece(file_name).ends_with(".png")) {
		image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
			DecodePng::Channels(wanted_channels));
	}
	else if (tensorflow::StringPiece(file_name).ends_with(".gif")) {
		// gif decoder returns 4-D tensor, remove the first dim
		image_reader = Squeeze(root.WithOpName("squeeze_first_dim"),
			DecodeGif(root.WithOpName("gif_reader"),
				file_reader));
	}
	else {
		// Assume if it's neither a PNG nor a GIF then it must be a JPEG.
		image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
			DecodeJpeg::Channels(wanted_channels));
	}
	// Now cast the image data to float so we can do normal math on it.
	auto float_caster =
		Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);
	// The convention for image ops in TensorFlow is that all images are expected
	// to be in batches, so that they're four-dimensional arrays with indices of
	// [batch, height, width, channel]. Because we only have a single image, we
	// have to add a batch dimension of 1 to the start with ExpandDims().
	auto dims_expander = ExpandDims(root, float_caster, 0);

	// Bilinearly resize the image to fit the required dimensions.
	auto resized = ResizeBilinear(
		root, dims_expander,
		Const(root.WithOpName("size"), { input_height, input_width }));
	// Subtract the mean and divide by the scale.
	Div(root.WithOpName(output_name), Sub(root, resized, { input_mean }),
	{ input_std });

	// This runs the GraphDef network definition that we've just constructed, and
	// returns the results in the output tensor.
	tensorflow::GraphDef graph;
	TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

	std::unique_ptr<tensorflow::Session> session(
		tensorflow::NewSession(tensorflow::SessionOptions()));
	TF_RETURN_IF_ERROR(session->Create(graph));
	TF_RETURN_IF_ERROR(session->Run({}, { output_name }, {}, out_tensors));
	return Status::OK();
}

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.

// 从硬盘上读取固化后的模型graph
Status LoadGraph(const string& graph_file_name,
	std::unique_ptr<tensorflow::Session>* session) {
	tensorflow::GraphDef graph_def;
	Status load_graph_status =
		ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
	if (!load_graph_status.ok()) {
		return tensorflow::errors::NotFound("Failed to load compute graph at '",
			graph_file_name, "'");
	}
	session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));

	Status session_create_status = (*session)->Create(graph_def);
	if (!session_create_status.ok()) {
		return session_create_status;
	}
	return Status::OK();
}

// Analyzes the output of the Inception graph to retrieve the highest scores and
// their positions in the tensor, which correspond to categories.
Status GetTopLabels(const std::vector<Tensor>& outputs, int how_many_labels,
	Tensor* indices, Tensor* scores) {
	auto root = tensorflow::Scope::NewRootScope();
	using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

	string output_name = "top_k";
	TopK(root.WithOpName(output_name), outputs[0], how_many_labels);
	// This runs the GraphDef network definition that we've just constructed, and
	// returns the results in the output tensors.
	tensorflow::GraphDef graph;
	TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

	std::unique_ptr<tensorflow::Session> session(
		tensorflow::NewSession(tensorflow::SessionOptions()));
	TF_RETURN_IF_ERROR(session->Create(graph));
	// The TopK node returns two outputs, the scores and their original indices,
	// so we have to append :0 and :1 to specify them both.
	std::vector<Tensor> out_tensors;
	TF_RETURN_IF_ERROR(session->Run({}, { output_name + ":0", output_name + ":1" },
	{}, &out_tensors));
	*scores = out_tensors[0];
	*indices = out_tensors[1];
	return Status::OK();
}

// Given the output of a model run, and the name of a file containing the labels
// this prints out the top five highest-scoring values.
Status PrintTopLabels(const std::vector<Tensor>& outputs,
	const string& labels_file_name) {
	std::vector<string> labels;
	size_t label_count;
	Status read_labels_status =
		ReadLabelsFile(labels_file_name, &labels, &label_count);
	if (!read_labels_status.ok()) {
		LOG(ERROR) << read_labels_status;
		return read_labels_status;
	}
	const int how_many_labels = std::min(5, static_cast<int>(label_count));
	Tensor indices;
	Tensor scores;
	TF_RETURN_IF_ERROR(GetTopLabels(outputs, how_many_labels, &indices, &scores));
	tensorflow::TTypes<float>::Flat scores_flat = scores.flat<float>();
	tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
	for (int pos = 0; pos < how_many_labels; ++pos) {
		const int label_index = indices_flat(pos);
		const float score = scores_flat(pos);
		LOG(INFO) << labels[label_index] << " (" << label_index << "): " << score;
	}
	return Status::OK();
}

// This is a testing function that returns whether the top label index is the
// one that's expected.
Status CheckTopLabel(const std::vector<Tensor>& outputs, int expected,
	bool* is_expected) {
	*is_expected = false;
	Tensor indices;
	Tensor scores;
	const int how_many_labels = 1;
	TF_RETURN_IF_ERROR(GetTopLabels(outputs, how_many_labels, &indices, &scores));
	tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
	if (indices_flat(0) != expected) {
		LOG(ERROR) << "Expected label #" << expected << " but got #"
			<< indices_flat(0);
		*is_expected = false;
	}
	else {
		*is_expected = true;
	}
	return Status::OK();
}

std::vector<string> SaveVideoToImages(const char *videofile, string imagePath, bool showProgress=false) {
	/* code for use OpenCV */
	// char *videofile = "C:\\dev\\tf_opencv\\bin\\model\\1.trim.264"; //"D:\\AVCaptures\\telus_scutter_near_end.mp4"; 
	vector<string> ret;

	// check if files already exists
	bool exists = false;
	for (auto &p : fs::directory_iterator(imagePath)) {
		ret.push_back(p.path().string());
		exists = true;
	}

	if (exists) {
		return ret;
	}

	VideoCapture capture(0);
	bool readsuccess = capture.open(videofile);

	if (!readsuccess) {
		cout << "open mp4 file error";
		return ret;
	}

	int debugFrame = 100, debugIndex = 0;
	if (capture.isOpened()) {
		char c;
		if (showProgress) {
			namedWindow("Video", 0);
			namedWindow("OutVideo", 1);
		}
		Mat imageMat, labeledMat;
		stringstream ss;
		while (true) {
			bool readSuccess = capture.read(imageMat);
			if (!readSuccess) {
				cout << "Done" << endl;
				break;
			}
			if (showProgress) {
				imshow("Video", imageMat);
			}
			ss.str("");
			ss << imagePath << debugIndex << ".jpg";
			string jpg = ss.str();
			imwrite(jpg, imageMat);
			ret.push_back(jpg);
			labeledMat = imageMat.clone();
			if (showProgress) {
				imshow("OutVideo", labeledMat);
			}

			c = waitKey(100) & 0xFF;
			if (c == 'ESC' || debugIndex++ > debugFrame) {
				break;
			}
		}
	}

	return ret;
}

int GetFloorNumber(string imgFilePath, std::unique_ptr<tensorflow::Session> *session, int how_many_labels = 10, int input_height = 224, int input_width = 224, int input_mean = 0, int input_std = 255) {
	// int input_height = 224, input_width = 224, input_mean = 0, input_std = 255;
	string input_layer = "data_node", output_layer = "Softmax_1";
	std::vector<Tensor> resized_tensors;
	std::vector<Tensor> outputs;

	Status read_tensor_status = ReadTensorFromImageFile(imgFilePath, input_height, input_width, input_mean,
		input_std, &resized_tensors);

	if (!read_tensor_status.ok()) {
		LOG(ERROR) << read_tensor_status;
		return -1;
	}
	const Tensor& resized_tensor = resized_tensors[0];

	Status run_status = (*session)->Run({ { input_layer, resized_tensor } }, { output_layer }, {}, &outputs);

	Tensor indices;
	Tensor scores;
	int ret = -1;
	float maxPossible = -1.0;
	GetTopLabels(outputs, how_many_labels, &indices, &scores);
	tensorflow::TTypes<float>::Flat scores_flat = scores.flat<float>();
	tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
	for (int pos = 0; pos < how_many_labels; ++pos) {
		const int label_index = indices_flat(pos);
		const float score = scores_flat(pos);
		LOG(INFO) << label_index << " (" << label_index << "): " << score;
		if (score > maxPossible) {
			maxPossible = score;
			ret = label_index;
		}
	}

	return ret;
}

int GetArrowNumber(string imgFilePath, std::unique_ptr<tensorflow::Session> *session) {
	return GetFloorNumber(imgFilePath, session, 2);
}

Status Read_Classfier(string imgFilePath, std::unique_ptr<tensorflow::Session> *session) {
	// https://52.175.242.176:8000/user/stonepeter/notebooks/deephead_tf/use_frozen_pb.ipynb
	std::vector<string> input_layers = { "input_6","input_5" };
	// https://github.com/peter6888/keras-frcnn/blob/master/test.ipynb
	std::vector<string> output_layers = { "dense_class_2/Reshape_1", "dense_regress_2/Reshape_1" };

	std::vector<Tensor> resized_tensors;
	std::vector<Tensor> outputs;

	// refs https://github.com/tensorflow/tensorflow/blob/38e0922d1e2dcd572379af4496f878492e9f689a/tensorflow/core/public/README.md
	// and tensorflow/tensorflow/core/public/session.h
	/*
	virtual Status Run(const std::vector<std::pair<string, Tensor> >& inputs,
	const std::vector<string>& output_tensor_names,
	const std::vector<string>& target_node_names,
	std::vector<Tensor>* outputs) = 0;
	*/
	return Status::OK();
}

Status Read_RPN(string imgFilePath, std::unique_ptr<tensorflow::Session> *session) {
	return Status::OK();
}


int main(int argc, char* argv[]) {
	vector<string> images = SaveVideoToImages("C:\\dev\\tf_opencv\\bin\\model\\1.trim.264",
		"C:\\dev\\tf_opencv\\bin\\model\\frames\\");
	// These are the command-line flags the program can understand.
	// They define where the graph and input data is located, and what kind of
	// input the model expects. If you train your own model, or use something
	// other than inception_v3, then you'll need to update these.
	cv::Mat cvmat;
	string image = "0022.jpg";
	string graph = "frozen_model_classifier.pb";
	string labels = "imagenet_slim_labels.txt";
	int32 input_width = INPUT_WIDTH;
	int32 input_height = INPUT_HEIGHT;
	int32 input_mean = 0;
	int32 input_std = 255;
	string input_layer = "input";
	string output_layer = "InceptionV3/Predictions/Reshape_1";
	bool self_test = false;
	string root_dir = "C:\\dev\\tf_opencv\\bin\\model";
	std::vector<Flag> flag_list = {
		Flag("image", &image, "image to be processed"),
		Flag("graph", &graph, "graph to be executed"),
		Flag("labels", &labels, "name of file containing labels"),
		Flag("input_width", &input_width, "resize image to this width in pixels"),
		Flag("input_height", &input_height, "resize image to this height in pixels"),
		Flag("input_mean", &input_mean, "scale pixel values to this mean"),
		Flag("input_std", &input_std, "scale pixel values to this std deviation"),
		Flag("input_layer", &input_layer, "name of image input layer"),
		Flag("output_layer", &output_layer, "name of output layer"),
		Flag("self_test", &self_test, "run a self test"),
		Flag("root_dir", &root_dir,
		"interpret image and graph file names relative to this directory"),
	};
	string usage = tensorflow::Flags::Usage(argv[0], flag_list);
	const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
	if (!parse_result) {
		LOG(ERROR) << usage;
		return -1;
	}

	// We need to call this to set up global state for TensorFlow.
	tensorflow::port::InitMain(argv[0], &argc, &argv);
	if (argc > 1) {
		LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
		return -1;
	}

	// First we load and initialize the model.
	std::unique_ptr<tensorflow::Session> session;
	//string graph_path = tensorflow::io::JoinPath(root_dir, graph);
	//string label_path = tensorflow::io::JoinPath(root_dir, labels);

	string floor_graph = "frozen_model_floor.pb";
	string floor_graph_path = tensorflow::io::JoinPath(root_dir, floor_graph);

	Status load_graph_status = LoadGraph(floor_graph_path, &session);
	if (!load_graph_status.ok()) {
		LOG(ERROR) << load_graph_status;
		return -1;
	}

	std::unique_ptr<tensorflow::Session> session_arrow;
	string arrow_graph = "frozen_model_arrow.pb";
	string arrow_graph_path = tensorflow::io::JoinPath(root_dir, arrow_graph);
	load_graph_status = LoadGraph(arrow_graph_path, &session_arrow);
	if (!load_graph_status.ok()) {
		LOG(ERROR) << load_graph_status;
		return -1;
	}

	std::unique_ptr<tensorflow::Session> session_rpn;
	string rpn_graph = "frozen_model_rpn.pb";
	string rpn_graph_path = tensorflow::io::JoinPath(root_dir, rpn_graph);
	load_graph_status = LoadGraph(rpn_graph_path, &session_rpn);
	if (!load_graph_status.ok()) {
		LOG(ERROR) << load_graph_status;
		return -1;
	}

	std::unique_ptr<tensorflow::Session> session_classifier;
	string classifier_graph = "frozen_model_classifier.pb";
	string classifier_graph_path = tensorflow::io::JoinPath(root_dir, classifier_graph);
	load_graph_status = LoadGraph(classifier_graph_path, &session_classifier);
	if (!load_graph_status.ok()) {
		LOG(ERROR) << load_graph_status;
		return -1;
	}

	for (string image_path : images) {
		cout << "for image:" << image_path << " floor " << GetFloorNumber(image_path, &session) << endl;
		cout << "arrow " << GetArrowNumber(image_path, &session_arrow) << endl;
		//Status read_tensor_status = ReadTensorFromImageFile(image_path, input_height, input_width, input_mean,
		//	input_std, &resized_tensors);

		//if (!read_tensor_status.ok()) {
		//	LOG(ERROR) << read_tensor_status;
		//	return -1;
		//}
		//const Tensor& resized_tensor = resized_tensors[0];

		//clockBegin = clock();
		//Status run_status = session->Run({ { input_layer, resized_tensor } }, { output_layer }, {}, &outputs);
		//clockEnd = clock();

		//std::cout << run_status.ToString() << std::endl;
		//std::cout << "time consume: " << clockEnd - clockBegin << "ms" << std::endl;
		////std::cout << "output_size:" << outputs.size() << std::endl;

		//// This is for automated testing to make sure we get the expected result with
		//// the default settings. We know that label 653 (military uniform) should be
		//// the top label for the Admiral Hopper image.
		//if (self_test) {
		//	bool expected_matches;
		//	Status check_status = CheckTopLabel(outputs, 653, &expected_matches);
		//	if (!check_status.ok()) {
		//		LOG(ERROR) << "Running check failed: " << check_status;
		//		return -1;
		//	}
		//	if (!expected_matches) {
		//		LOG(ERROR) << "Self-test failed!";
		//		return -1;
		//	}
		//}

		//// Do something interesting with the results we've generated.
		//Status print_status = PrintTopLabels(outputs, label_path);
		//if (!print_status.ok()) {
		//	LOG(ERROR) << "Running print failed: " << print_status;
		//	return -1;
		//}
	}
	return 0;
}
