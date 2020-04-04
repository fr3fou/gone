// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.20.1
// 	protoc        (unknown)
// source: gone.proto

package pb

import (
	proto "github.com/golang/protobuf/proto"
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	reflect "reflect"
	sync "sync"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

// This is a compile-time assertion that a sufficiently up-to-date version
// of the legacy proto package is being used.
const _ = proto.ProtoPackageIsVersion4

type NeuralNetwork struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Weights      []*Matrix `protobuf:"bytes,1,rep,name=Weights,json=weights,proto3" json:"Weights,omitempty"`
	Biases       []*Matrix `protobuf:"bytes,2,rep,name=Biases,json=biases,proto3" json:"Biases,omitempty"`
	Activations  []*Matrix `protobuf:"bytes,3,rep,name=Activations,json=activations,proto3" json:"Activations,omitempty"`
	LearningRate float64   `protobuf:"fixed64,4,opt,name=LearningRate,json=learningRate,proto3" json:"LearningRate,omitempty"`
	Layers       []*Layer  `protobuf:"bytes,5,rep,name=Layers,json=layers,proto3" json:"Layers,omitempty"`
	DebugMode    bool      `protobuf:"varint,6,opt,name=DebugMode,json=debugMode,proto3" json:"DebugMode,omitempty"`
	Loss         string    `protobuf:"bytes,7,opt,name=Loss,json=loss,proto3" json:"Loss,omitempty"`
}

func (x *NeuralNetwork) Reset() {
	*x = NeuralNetwork{}
	if protoimpl.UnsafeEnabled {
		mi := &file_gone_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *NeuralNetwork) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*NeuralNetwork) ProtoMessage() {}

func (x *NeuralNetwork) ProtoReflect() protoreflect.Message {
	mi := &file_gone_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use NeuralNetwork.ProtoReflect.Descriptor instead.
func (*NeuralNetwork) Descriptor() ([]byte, []int) {
	return file_gone_proto_rawDescGZIP(), []int{0}
}

func (x *NeuralNetwork) GetWeights() []*Matrix {
	if x != nil {
		return x.Weights
	}
	return nil
}

func (x *NeuralNetwork) GetBiases() []*Matrix {
	if x != nil {
		return x.Biases
	}
	return nil
}

func (x *NeuralNetwork) GetActivations() []*Matrix {
	if x != nil {
		return x.Activations
	}
	return nil
}

func (x *NeuralNetwork) GetLearningRate() float64 {
	if x != nil {
		return x.LearningRate
	}
	return 0
}

func (x *NeuralNetwork) GetLayers() []*Layer {
	if x != nil {
		return x.Layers
	}
	return nil
}

func (x *NeuralNetwork) GetDebugMode() bool {
	if x != nil {
		return x.DebugMode
	}
	return false
}

func (x *NeuralNetwork) GetLoss() string {
	if x != nil {
		return x.Loss
	}
	return ""
}

type Layer struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Nodes     int32  `protobuf:"varint,1,opt,name=Nodes,json=nodes,proto3" json:"Nodes,omitempty"`
	Activator string `protobuf:"bytes,2,opt,name=Activator,json=activator,proto3" json:"Activator,omitempty"` // TODO: store the struct along with configuration
}

func (x *Layer) Reset() {
	*x = Layer{}
	if protoimpl.UnsafeEnabled {
		mi := &file_gone_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Layer) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Layer) ProtoMessage() {}

func (x *Layer) ProtoReflect() protoreflect.Message {
	mi := &file_gone_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Layer.ProtoReflect.Descriptor instead.
func (*Layer) Descriptor() ([]byte, []int) {
	return file_gone_proto_rawDescGZIP(), []int{1}
}

func (x *Layer) GetNodes() int32 {
	if x != nil {
		return x.Nodes
	}
	return 0
}

func (x *Layer) GetActivator() string {
	if x != nil {
		return x.Activator
	}
	return ""
}

type Matrix struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Rows    int32     `protobuf:"varint,1,opt,name=Rows,json=rows,proto3" json:"Rows,omitempty"`
	Columns int32     `protobuf:"varint,2,opt,name=Columns,json=columns,proto3" json:"Columns,omitempty"`
	Data    []float64 `protobuf:"fixed64,3,rep,packed,name=Data,json=data,proto3" json:"Data,omitempty"` // flattened 2D array
}

func (x *Matrix) Reset() {
	*x = Matrix{}
	if protoimpl.UnsafeEnabled {
		mi := &file_gone_proto_msgTypes[2]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Matrix) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Matrix) ProtoMessage() {}

func (x *Matrix) ProtoReflect() protoreflect.Message {
	mi := &file_gone_proto_msgTypes[2]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Matrix.ProtoReflect.Descriptor instead.
func (*Matrix) Descriptor() ([]byte, []int) {
	return file_gone_proto_rawDescGZIP(), []int{2}
}

func (x *Matrix) GetRows() int32 {
	if x != nil {
		return x.Rows
	}
	return 0
}

func (x *Matrix) GetColumns() int32 {
	if x != nil {
		return x.Columns
	}
	return 0
}

func (x *Matrix) GetData() []float64 {
	if x != nil {
		return x.Data
	}
	return nil
}

var File_gone_proto protoreflect.FileDescriptor

var file_gone_proto_rawDesc = []byte{
	0x0a, 0x0a, 0x67, 0x6f, 0x6e, 0x65, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x12, 0x02, 0x70, 0x62,
	0x22, 0x80, 0x02, 0x0a, 0x0d, 0x4e, 0x65, 0x75, 0x72, 0x61, 0x6c, 0x4e, 0x65, 0x74, 0x77, 0x6f,
	0x72, 0x6b, 0x12, 0x24, 0x0a, 0x07, 0x57, 0x65, 0x69, 0x67, 0x68, 0x74, 0x73, 0x18, 0x01, 0x20,
	0x03, 0x28, 0x0b, 0x32, 0x0a, 0x2e, 0x70, 0x62, 0x2e, 0x4d, 0x61, 0x74, 0x72, 0x69, 0x78, 0x52,
	0x07, 0x77, 0x65, 0x69, 0x67, 0x68, 0x74, 0x73, 0x12, 0x22, 0x0a, 0x06, 0x42, 0x69, 0x61, 0x73,
	0x65, 0x73, 0x18, 0x02, 0x20, 0x03, 0x28, 0x0b, 0x32, 0x0a, 0x2e, 0x70, 0x62, 0x2e, 0x4d, 0x61,
	0x74, 0x72, 0x69, 0x78, 0x52, 0x06, 0x62, 0x69, 0x61, 0x73, 0x65, 0x73, 0x12, 0x2c, 0x0a, 0x0b,
	0x41, 0x63, 0x74, 0x69, 0x76, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x73, 0x18, 0x03, 0x20, 0x03, 0x28,
	0x0b, 0x32, 0x0a, 0x2e, 0x70, 0x62, 0x2e, 0x4d, 0x61, 0x74, 0x72, 0x69, 0x78, 0x52, 0x0b, 0x61,
	0x63, 0x74, 0x69, 0x76, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x73, 0x12, 0x22, 0x0a, 0x0c, 0x4c, 0x65,
	0x61, 0x72, 0x6e, 0x69, 0x6e, 0x67, 0x52, 0x61, 0x74, 0x65, 0x18, 0x04, 0x20, 0x01, 0x28, 0x01,
	0x52, 0x0c, 0x6c, 0x65, 0x61, 0x72, 0x6e, 0x69, 0x6e, 0x67, 0x52, 0x61, 0x74, 0x65, 0x12, 0x21,
	0x0a, 0x06, 0x4c, 0x61, 0x79, 0x65, 0x72, 0x73, 0x18, 0x05, 0x20, 0x03, 0x28, 0x0b, 0x32, 0x09,
	0x2e, 0x70, 0x62, 0x2e, 0x4c, 0x61, 0x79, 0x65, 0x72, 0x52, 0x06, 0x6c, 0x61, 0x79, 0x65, 0x72,
	0x73, 0x12, 0x1c, 0x0a, 0x09, 0x44, 0x65, 0x62, 0x75, 0x67, 0x4d, 0x6f, 0x64, 0x65, 0x18, 0x06,
	0x20, 0x01, 0x28, 0x08, 0x52, 0x09, 0x64, 0x65, 0x62, 0x75, 0x67, 0x4d, 0x6f, 0x64, 0x65, 0x12,
	0x12, 0x0a, 0x04, 0x4c, 0x6f, 0x73, 0x73, 0x18, 0x07, 0x20, 0x01, 0x28, 0x09, 0x52, 0x04, 0x6c,
	0x6f, 0x73, 0x73, 0x22, 0x3b, 0x0a, 0x05, 0x4c, 0x61, 0x79, 0x65, 0x72, 0x12, 0x14, 0x0a, 0x05,
	0x4e, 0x6f, 0x64, 0x65, 0x73, 0x18, 0x01, 0x20, 0x01, 0x28, 0x05, 0x52, 0x05, 0x6e, 0x6f, 0x64,
	0x65, 0x73, 0x12, 0x1c, 0x0a, 0x09, 0x41, 0x63, 0x74, 0x69, 0x76, 0x61, 0x74, 0x6f, 0x72, 0x18,
	0x02, 0x20, 0x01, 0x28, 0x09, 0x52, 0x09, 0x61, 0x63, 0x74, 0x69, 0x76, 0x61, 0x74, 0x6f, 0x72,
	0x22, 0x4a, 0x0a, 0x06, 0x4d, 0x61, 0x74, 0x72, 0x69, 0x78, 0x12, 0x12, 0x0a, 0x04, 0x52, 0x6f,
	0x77, 0x73, 0x18, 0x01, 0x20, 0x01, 0x28, 0x05, 0x52, 0x04, 0x72, 0x6f, 0x77, 0x73, 0x12, 0x18,
	0x0a, 0x07, 0x43, 0x6f, 0x6c, 0x75, 0x6d, 0x6e, 0x73, 0x18, 0x02, 0x20, 0x01, 0x28, 0x05, 0x52,
	0x07, 0x63, 0x6f, 0x6c, 0x75, 0x6d, 0x6e, 0x73, 0x12, 0x12, 0x0a, 0x04, 0x44, 0x61, 0x74, 0x61,
	0x18, 0x03, 0x20, 0x03, 0x28, 0x01, 0x52, 0x04, 0x64, 0x61, 0x74, 0x61, 0x42, 0x06, 0x5a, 0x04,
	0x2e, 0x3b, 0x70, 0x62, 0x62, 0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_gone_proto_rawDescOnce sync.Once
	file_gone_proto_rawDescData = file_gone_proto_rawDesc
)

func file_gone_proto_rawDescGZIP() []byte {
	file_gone_proto_rawDescOnce.Do(func() {
		file_gone_proto_rawDescData = protoimpl.X.CompressGZIP(file_gone_proto_rawDescData)
	})
	return file_gone_proto_rawDescData
}

var file_gone_proto_msgTypes = make([]protoimpl.MessageInfo, 3)
var file_gone_proto_goTypes = []interface{}{
	(*NeuralNetwork)(nil), // 0: pb.NeuralNetwork
	(*Layer)(nil),         // 1: pb.Layer
	(*Matrix)(nil),        // 2: pb.Matrix
}
var file_gone_proto_depIdxs = []int32{
	2, // 0: pb.NeuralNetwork.Weights:type_name -> pb.Matrix
	2, // 1: pb.NeuralNetwork.Biases:type_name -> pb.Matrix
	2, // 2: pb.NeuralNetwork.Activations:type_name -> pb.Matrix
	1, // 3: pb.NeuralNetwork.Layers:type_name -> pb.Layer
	4, // [4:4] is the sub-list for method output_type
	4, // [4:4] is the sub-list for method input_type
	4, // [4:4] is the sub-list for extension type_name
	4, // [4:4] is the sub-list for extension extendee
	0, // [0:4] is the sub-list for field type_name
}

func init() { file_gone_proto_init() }
func file_gone_proto_init() {
	if File_gone_proto != nil {
		return
	}
	if !protoimpl.UnsafeEnabled {
		file_gone_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*NeuralNetwork); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_gone_proto_msgTypes[1].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Layer); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_gone_proto_msgTypes[2].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Matrix); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_gone_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   3,
			NumExtensions: 0,
			NumServices:   0,
		},
		GoTypes:           file_gone_proto_goTypes,
		DependencyIndexes: file_gone_proto_depIdxs,
		MessageInfos:      file_gone_proto_msgTypes,
	}.Build()
	File_gone_proto = out.File
	file_gone_proto_rawDesc = nil
	file_gone_proto_goTypes = nil
	file_gone_proto_depIdxs = nil
}
