┘ё
жЎ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
В
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
┴
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8ош
~
Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/v
w
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes
:*
dtype0
З
Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*&
shared_nameAdam/dense_3/kernel/v
А
)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
_output_shapes
:	А*
dtype0

Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_nameAdam/dense_2/bias/v
x
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes	
:А*
dtype0
И
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А@А*&
shared_nameAdam/dense_2/kernel/v
Б
)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v* 
_output_shapes
:
А@А*
dtype0
~
Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/m
w
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes
:*
dtype0
З
Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*&
shared_nameAdam/dense_3/kernel/m
А
)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes
:	А*
dtype0

Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_nameAdam/dense_2/bias/m
x
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes	
:А*
dtype0
И
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А@А*&
shared_nameAdam/dense_2/kernel/m
Б
)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m* 
_output_shapes
:
А@А*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
{
block5_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameblock5_conv3/bias
t
%block5_conv3/bias/Read/ReadVariableOpReadVariableOpblock5_conv3/bias*
_output_shapes	
:А*
dtype0
М
block5_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*$
shared_nameblock5_conv3/kernel
Е
'block5_conv3/kernel/Read/ReadVariableOpReadVariableOpblock5_conv3/kernel*(
_output_shapes
:АА*
dtype0
{
block5_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameblock5_conv2/bias
t
%block5_conv2/bias/Read/ReadVariableOpReadVariableOpblock5_conv2/bias*
_output_shapes	
:А*
dtype0
М
block5_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*$
shared_nameblock5_conv2/kernel
Е
'block5_conv2/kernel/Read/ReadVariableOpReadVariableOpblock5_conv2/kernel*(
_output_shapes
:АА*
dtype0
{
block5_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameblock5_conv1/bias
t
%block5_conv1/bias/Read/ReadVariableOpReadVariableOpblock5_conv1/bias*
_output_shapes	
:А*
dtype0
М
block5_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*$
shared_nameblock5_conv1/kernel
Е
'block5_conv1/kernel/Read/ReadVariableOpReadVariableOpblock5_conv1/kernel*(
_output_shapes
:АА*
dtype0
{
block4_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameblock4_conv3/bias
t
%block4_conv3/bias/Read/ReadVariableOpReadVariableOpblock4_conv3/bias*
_output_shapes	
:А*
dtype0
М
block4_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*$
shared_nameblock4_conv3/kernel
Е
'block4_conv3/kernel/Read/ReadVariableOpReadVariableOpblock4_conv3/kernel*(
_output_shapes
:АА*
dtype0
{
block4_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameblock4_conv2/bias
t
%block4_conv2/bias/Read/ReadVariableOpReadVariableOpblock4_conv2/bias*
_output_shapes	
:А*
dtype0
М
block4_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*$
shared_nameblock4_conv2/kernel
Е
'block4_conv2/kernel/Read/ReadVariableOpReadVariableOpblock4_conv2/kernel*(
_output_shapes
:АА*
dtype0
{
block4_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameblock4_conv1/bias
t
%block4_conv1/bias/Read/ReadVariableOpReadVariableOpblock4_conv1/bias*
_output_shapes	
:А*
dtype0
М
block4_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*$
shared_nameblock4_conv1/kernel
Е
'block4_conv1/kernel/Read/ReadVariableOpReadVariableOpblock4_conv1/kernel*(
_output_shapes
:АА*
dtype0
{
block3_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameblock3_conv3/bias
t
%block3_conv3/bias/Read/ReadVariableOpReadVariableOpblock3_conv3/bias*
_output_shapes	
:А*
dtype0
М
block3_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*$
shared_nameblock3_conv3/kernel
Е
'block3_conv3/kernel/Read/ReadVariableOpReadVariableOpblock3_conv3/kernel*(
_output_shapes
:АА*
dtype0
{
block3_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameblock3_conv2/bias
t
%block3_conv2/bias/Read/ReadVariableOpReadVariableOpblock3_conv2/bias*
_output_shapes	
:А*
dtype0
М
block3_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*$
shared_nameblock3_conv2/kernel
Е
'block3_conv2/kernel/Read/ReadVariableOpReadVariableOpblock3_conv2/kernel*(
_output_shapes
:АА*
dtype0
{
block3_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameblock3_conv1/bias
t
%block3_conv1/bias/Read/ReadVariableOpReadVariableOpblock3_conv1/bias*
_output_shapes	
:А*
dtype0
М
block3_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*$
shared_nameblock3_conv1/kernel
Е
'block3_conv1/kernel/Read/ReadVariableOpReadVariableOpblock3_conv1/kernel*(
_output_shapes
:АА*
dtype0
{
block2_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameblock2_conv2/bias
t
%block2_conv2/bias/Read/ReadVariableOpReadVariableOpblock2_conv2/bias*
_output_shapes	
:А*
dtype0
М
block2_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:АА*$
shared_nameblock2_conv2/kernel
Е
'block2_conv2/kernel/Read/ReadVariableOpReadVariableOpblock2_conv2/kernel*(
_output_shapes
:АА*
dtype0
{
block2_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_nameblock2_conv1/bias
t
%block2_conv1/bias/Read/ReadVariableOpReadVariableOpblock2_conv1/bias*
_output_shapes	
:А*
dtype0
Л
block2_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*$
shared_nameblock2_conv1/kernel
Д
'block2_conv1/kernel/Read/ReadVariableOpReadVariableOpblock2_conv1/kernel*'
_output_shapes
:@А*
dtype0
z
block1_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameblock1_conv2/bias
s
%block1_conv2/bias/Read/ReadVariableOpReadVariableOpblock1_conv2/bias*
_output_shapes
:@*
dtype0
К
block1_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*$
shared_nameblock1_conv2/kernel
Г
'block1_conv2/kernel/Read/ReadVariableOpReadVariableOpblock1_conv2/kernel*&
_output_shapes
:@@*
dtype0
z
block1_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameblock1_conv1/bias
s
%block1_conv1/bias/Read/ReadVariableOpReadVariableOpblock1_conv1/bias*
_output_shapes
:@*
dtype0
К
block1_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameblock1_conv1/kernel
Г
'block1_conv1/kernel/Read/ReadVariableOpReadVariableOpblock1_conv1/kernel*&
_output_shapes
:@*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	А*
dtype0
q
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:А*
dtype0
z
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А@А*
shared_namedense_2/kernel
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
А@А*
dtype0
Т
serving_default_vgg16_inputPlaceholder*1
_output_shapes
:         ЦЦ*
dtype0*&
shape:         ЦЦ
╕
StatefulPartitionedCallStatefulPartitionedCallserving_default_vgg16_inputblock1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock5_conv1/kernelblock5_conv1/biasblock5_conv2/kernelblock5_conv2/biasblock5_conv3/kernelblock5_conv3/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8В *,
f'R%
#__inference_signature_wrapper_18998

NoOpNoOp
чЬ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*бЬ
valueЦЬBТЬ BКЬ
╬
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
х
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
layer_with_weights-5
layer-8
layer_with_weights-6
layer-9
layer-10
layer_with_weights-7
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer-14
layer_with_weights-10
layer-15
layer_with_weights-11
layer-16
layer_with_weights-12
layer-17
 layer-18
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses*
О
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses* 
ж
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

3kernel
4bias*
ж
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

;kernel
<bias*
ъ
=0
>1
?2
@3
A4
B5
C6
D7
E8
F9
G10
H11
I12
J13
K14
L15
M16
N17
O18
P19
Q20
R21
S22
T23
U24
V25
326
427
;28
<29*
 
30
41
;2
<3*
* 
░
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
6
\trace_0
]trace_1
^trace_2
_trace_3* 
6
`trace_0
atrace_1
btrace_2
ctrace_3* 
* 
Ф
diter

ebeta_1

fbeta_2
	gdecay
hlearning_rate3mО4mП;mР<mС3vТ4vУ;vФ<vХ*

iserving_default* 
* 
╚
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses

=kernel
>bias
 p_jit_compiled_convolution_op*
╚
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses

?kernel
@bias
 w_jit_compiled_convolution_op*
О
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses* 
═
~	variables
trainable_variables
Аregularization_losses
Б	keras_api
В__call__
+Г&call_and_return_all_conditional_losses

Akernel
Bbias
!Д_jit_compiled_convolution_op*
╧
Е	variables
Жtrainable_variables
Зregularization_losses
И	keras_api
Й__call__
+К&call_and_return_all_conditional_losses

Ckernel
Dbias
!Л_jit_compiled_convolution_op*
Ф
М	variables
Нtrainable_variables
Оregularization_losses
П	keras_api
Р__call__
+С&call_and_return_all_conditional_losses* 
╧
Т	variables
Уtrainable_variables
Фregularization_losses
Х	keras_api
Ц__call__
+Ч&call_and_return_all_conditional_losses

Ekernel
Fbias
!Ш_jit_compiled_convolution_op*
╧
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Ь	keras_api
Э__call__
+Ю&call_and_return_all_conditional_losses

Gkernel
Hbias
!Я_jit_compiled_convolution_op*
╧
а	variables
бtrainable_variables
вregularization_losses
г	keras_api
д__call__
+е&call_and_return_all_conditional_losses

Ikernel
Jbias
!ж_jit_compiled_convolution_op*
Ф
з	variables
иtrainable_variables
йregularization_losses
к	keras_api
л__call__
+м&call_and_return_all_conditional_losses* 
╧
н	variables
оtrainable_variables
пregularization_losses
░	keras_api
▒__call__
+▓&call_and_return_all_conditional_losses

Kkernel
Lbias
!│_jit_compiled_convolution_op*
╧
┤	variables
╡trainable_variables
╢regularization_losses
╖	keras_api
╕__call__
+╣&call_and_return_all_conditional_losses

Mkernel
Nbias
!║_jit_compiled_convolution_op*
╧
╗	variables
╝trainable_variables
╜regularization_losses
╛	keras_api
┐__call__
+└&call_and_return_all_conditional_losses

Okernel
Pbias
!┴_jit_compiled_convolution_op*
Ф
┬	variables
├trainable_variables
─regularization_losses
┼	keras_api
╞__call__
+╟&call_and_return_all_conditional_losses* 
╧
╚	variables
╔trainable_variables
╩regularization_losses
╦	keras_api
╠__call__
+═&call_and_return_all_conditional_losses

Qkernel
Rbias
!╬_jit_compiled_convolution_op*
╧
╧	variables
╨trainable_variables
╤regularization_losses
╥	keras_api
╙__call__
+╘&call_and_return_all_conditional_losses

Skernel
Tbias
!╒_jit_compiled_convolution_op*
╧
╓	variables
╫trainable_variables
╪regularization_losses
┘	keras_api
┌__call__
+█&call_and_return_all_conditional_losses

Ukernel
Vbias
!▄_jit_compiled_convolution_op*
Ф
▌	variables
▐trainable_variables
▀regularization_losses
р	keras_api
с__call__
+т&call_and_return_all_conditional_losses* 
╩
=0
>1
?2
@3
A4
B5
C6
D7
E8
F9
G10
H11
I12
J13
K14
L15
M16
N17
O18
P19
Q20
R21
S22
T23
U24
V25*
* 
* 
Ш
уnon_trainable_variables
фlayers
хmetrics
 цlayer_regularization_losses
чlayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*
:
шtrace_0
щtrace_1
ъtrace_2
ыtrace_3* 
:
ьtrace_0
эtrace_1
юtrace_2
яtrace_3* 
* 
* 
* 
Ц
Ёnon_trainable_variables
ёlayers
Єmetrics
 єlayer_regularization_losses
Їlayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses* 

їtrace_0* 

Ўtrace_0* 

30
41*

30
41*
* 
Ш
ўnon_trainable_variables
°layers
∙metrics
 ·layer_regularization_losses
√layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses*

№trace_0* 

¤trace_0* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

;0
<1*

;0
<1*
* 
Ш
■non_trainable_variables
 layers
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*

Гtrace_0* 

Дtrace_0* 
^X
VARIABLE_VALUEdense_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEblock1_conv1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEblock1_conv1/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEblock1_conv2/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEblock1_conv2/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEblock2_conv1/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEblock2_conv1/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEblock2_conv2/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEblock2_conv2/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEblock3_conv1/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEblock3_conv1/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock3_conv2/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock3_conv2/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock3_conv3/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock3_conv3/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock4_conv1/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock4_conv1/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock4_conv2/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock4_conv2/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock4_conv3/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock4_conv3/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock5_conv1/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock5_conv1/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock5_conv2/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock5_conv2/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock5_conv3/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock5_conv3/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
╩
=0
>1
?2
@3
A4
B5
C6
D7
E8
F9
G10
H11
I12
J13
K14
L15
M16
N17
O18
P19
Q20
R21
S22
T23
U24
V25*
 
0
1
2
3*

Е0
Ж1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 

=0
>1*
* 
* 
Ш
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses*

Мtrace_0* 

Нtrace_0* 
* 

?0
@1*
* 
* 
Ш
Оnon_trainable_variables
Пlayers
Рmetrics
 Сlayer_regularization_losses
Тlayer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses*

Уtrace_0* 

Фtrace_0* 
* 
* 
* 
* 
Ц
Хnon_trainable_variables
Цlayers
Чmetrics
 Шlayer_regularization_losses
Щlayer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses* 

Ъtrace_0* 

Ыtrace_0* 

A0
B1*
* 
* 
Ь
Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
~	variables
trainable_variables
Аregularization_losses
В__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses*

бtrace_0* 

вtrace_0* 
* 

C0
D1*
* 
* 
Ю
гnon_trainable_variables
дlayers
еmetrics
 жlayer_regularization_losses
зlayer_metrics
Е	variables
Жtrainable_variables
Зregularization_losses
Й__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses*

иtrace_0* 

йtrace_0* 
* 
* 
* 
* 
Ь
кnon_trainable_variables
лlayers
мmetrics
 нlayer_regularization_losses
оlayer_metrics
М	variables
Нtrainable_variables
Оregularization_losses
Р__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses* 

пtrace_0* 

░trace_0* 

E0
F1*
* 
* 
Ю
▒non_trainable_variables
▓layers
│metrics
 ┤layer_regularization_losses
╡layer_metrics
Т	variables
Уtrainable_variables
Фregularization_losses
Ц__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses*

╢trace_0* 

╖trace_0* 
* 

G0
H1*
* 
* 
Ю
╕non_trainable_variables
╣layers
║metrics
 ╗layer_regularization_losses
╝layer_metrics
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Э__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses*

╜trace_0* 

╛trace_0* 
* 

I0
J1*
* 
* 
Ю
┐non_trainable_variables
└layers
┴metrics
 ┬layer_regularization_losses
├layer_metrics
а	variables
бtrainable_variables
вregularization_losses
д__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses*

─trace_0* 

┼trace_0* 
* 
* 
* 
* 
Ь
╞non_trainable_variables
╟layers
╚metrics
 ╔layer_regularization_losses
╩layer_metrics
з	variables
иtrainable_variables
йregularization_losses
л__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses* 

╦trace_0* 

╠trace_0* 

K0
L1*
* 
* 
Ю
═non_trainable_variables
╬layers
╧metrics
 ╨layer_regularization_losses
╤layer_metrics
н	variables
оtrainable_variables
пregularization_losses
▒__call__
+▓&call_and_return_all_conditional_losses
'▓"call_and_return_conditional_losses*

╥trace_0* 

╙trace_0* 
* 

M0
N1*
* 
* 
Ю
╘non_trainable_variables
╒layers
╓metrics
 ╫layer_regularization_losses
╪layer_metrics
┤	variables
╡trainable_variables
╢regularization_losses
╕__call__
+╣&call_and_return_all_conditional_losses
'╣"call_and_return_conditional_losses*

┘trace_0* 

┌trace_0* 
* 

O0
P1*
* 
* 
Ю
█non_trainable_variables
▄layers
▌metrics
 ▐layer_regularization_losses
▀layer_metrics
╗	variables
╝trainable_variables
╜regularization_losses
┐__call__
+└&call_and_return_all_conditional_losses
'└"call_and_return_conditional_losses*

рtrace_0* 

сtrace_0* 
* 
* 
* 
* 
Ь
тnon_trainable_variables
уlayers
фmetrics
 хlayer_regularization_losses
цlayer_metrics
┬	variables
├trainable_variables
─regularization_losses
╞__call__
+╟&call_and_return_all_conditional_losses
'╟"call_and_return_conditional_losses* 

чtrace_0* 

шtrace_0* 

Q0
R1*
* 
* 
Ю
щnon_trainable_variables
ъlayers
ыmetrics
 ьlayer_regularization_losses
эlayer_metrics
╚	variables
╔trainable_variables
╩regularization_losses
╠__call__
+═&call_and_return_all_conditional_losses
'═"call_and_return_conditional_losses*

юtrace_0* 

яtrace_0* 
* 

S0
T1*
* 
* 
Ю
Ёnon_trainable_variables
ёlayers
Єmetrics
 єlayer_regularization_losses
Їlayer_metrics
╧	variables
╨trainable_variables
╤regularization_losses
╙__call__
+╘&call_and_return_all_conditional_losses
'╘"call_and_return_conditional_losses*

їtrace_0* 

Ўtrace_0* 
* 

U0
V1*
* 
* 
Ю
ўnon_trainable_variables
°layers
∙metrics
 ·layer_regularization_losses
√layer_metrics
╓	variables
╫trainable_variables
╪regularization_losses
┌__call__
+█&call_and_return_all_conditional_losses
'█"call_and_return_conditional_losses*

№trace_0* 

¤trace_0* 
* 
* 
* 
* 
Ь
■non_trainable_variables
 layers
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
▌	variables
▐trainable_variables
▀regularization_losses
с__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses* 

Гtrace_0* 

Дtrace_0* 
╩
=0
>1
?2
@3
A4
B5
C6
D7
E8
F9
G10
H11
I12
J13
K14
L15
M16
N17
O18
P19
Q20
R21
S22
T23
U24
V25*
Т
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
 18*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
Е	variables
Ж	keras_api

Зtotal

Иcount*
M
Й	variables
К	keras_api

Лtotal

Мcount
Н
_fn_kwargs*

=0
>1*
* 
* 
* 
* 
* 
* 

?0
@1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

A0
B1*
* 
* 
* 
* 
* 
* 

C0
D1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

E0
F1*
* 
* 
* 
* 
* 
* 

G0
H1*
* 
* 
* 
* 
* 
* 

I0
J1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

K0
L1*
* 
* 
* 
* 
* 
* 

M0
N1*
* 
* 
* 
* 
* 
* 

O0
P1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Q0
R1*
* 
* 
* 
* 
* 
* 

S0
T1*
* 
* 
* 
* 
* 
* 

U0
V1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

З0
И1*

Е	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Л0
М1*

Й	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
Б{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdam/dense_3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Б{
VARIABLE_VALUEAdam/dense_3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
╤
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp'block1_conv1/kernel/Read/ReadVariableOp%block1_conv1/bias/Read/ReadVariableOp'block1_conv2/kernel/Read/ReadVariableOp%block1_conv2/bias/Read/ReadVariableOp'block2_conv1/kernel/Read/ReadVariableOp%block2_conv1/bias/Read/ReadVariableOp'block2_conv2/kernel/Read/ReadVariableOp%block2_conv2/bias/Read/ReadVariableOp'block3_conv1/kernel/Read/ReadVariableOp%block3_conv1/bias/Read/ReadVariableOp'block3_conv2/kernel/Read/ReadVariableOp%block3_conv2/bias/Read/ReadVariableOp'block3_conv3/kernel/Read/ReadVariableOp%block3_conv3/bias/Read/ReadVariableOp'block4_conv1/kernel/Read/ReadVariableOp%block4_conv1/bias/Read/ReadVariableOp'block4_conv2/kernel/Read/ReadVariableOp%block4_conv2/bias/Read/ReadVariableOp'block4_conv3/kernel/Read/ReadVariableOp%block4_conv3/bias/Read/ReadVariableOp'block5_conv1/kernel/Read/ReadVariableOp%block5_conv1/bias/Read/ReadVariableOp'block5_conv2/kernel/Read/ReadVariableOp%block5_conv2/bias/Read/ReadVariableOp'block5_conv3/kernel/Read/ReadVariableOp%block5_conv3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOpConst*<
Tin5
321	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *'
f"R 
__inference__traced_save_20199
а	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_2/kerneldense_2/biasdense_3/kerneldense_3/biasblock1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock5_conv1/kernelblock5_conv1/biasblock5_conv2/kernelblock5_conv2/biasblock5_conv3/kernelblock5_conv3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/dense_3/kernel/mAdam/dense_3/bias/mAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/dense_3/kernel/vAdam/dense_3/bias/v*;
Tin4
220*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__traced_restore_20350╔═
┤
E
)__inference_flatten_1_layer_call_fn_19679

inputs
identity│
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_18400a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
Т
Г
G__inference_block5_conv1_layer_call_and_return_conditional_losses_19985

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		АY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         		Аj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         		Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         		А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
л
╔	
G__inference_sequential_1_layer_call_and_return_conditional_losses_18925
vgg16_input%
vgg16_18860:@
vgg16_18862:@%
vgg16_18864:@@
vgg16_18866:@&
vgg16_18868:@А
vgg16_18870:	А'
vgg16_18872:АА
vgg16_18874:	А'
vgg16_18876:АА
vgg16_18878:	А'
vgg16_18880:АА
vgg16_18882:	А'
vgg16_18884:АА
vgg16_18886:	А'
vgg16_18888:АА
vgg16_18890:	А'
vgg16_18892:АА
vgg16_18894:	А'
vgg16_18896:АА
vgg16_18898:	А'
vgg16_18900:АА
vgg16_18902:	А'
vgg16_18904:АА
vgg16_18906:	А'
vgg16_18908:АА
vgg16_18910:	А!
dense_2_18914:
А@А
dense_2_18916:	А 
dense_3_18919:	А
dense_3_18921:
identityИвdense_2/StatefulPartitionedCallвdense_3/StatefulPartitionedCallвvgg16/StatefulPartitionedCall┌
vgg16/StatefulPartitionedCallStatefulPartitionedCallvgg16_inputvgg16_18860vgg16_18862vgg16_18864vgg16_18866vgg16_18868vgg16_18870vgg16_18872vgg16_18874vgg16_18876vgg16_18878vgg16_18880vgg16_18882vgg16_18884vgg16_18886vgg16_18888vgg16_18890vgg16_18892vgg16_18894vgg16_18896vgg16_18898vgg16_18900vgg16_18902vgg16_18904vgg16_18906vgg16_18908vgg16_18910*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_vgg16_layer_call_and_return_conditional_losses_18074▌
flatten_1/PartitionedCallPartitionedCall&vgg16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_18400Й
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_18914dense_2_18916*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_18413О
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_18919dense_3_18921*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_18430w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         к
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall^vgg16/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:         ЦЦ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2>
vgg16/StatefulPartitionedCallvgg16/StatefulPartitionedCall:^ Z
1
_output_shapes
:         ЦЦ
%
_user_specified_namevgg16_input
▐V
Л
@__inference_vgg16_layer_call_and_return_conditional_losses_18334
input_2,
block1_conv1_18263:@ 
block1_conv1_18265:@,
block1_conv2_18268:@@ 
block1_conv2_18270:@-
block2_conv1_18274:@А!
block2_conv1_18276:	А.
block2_conv2_18279:АА!
block2_conv2_18281:	А.
block3_conv1_18285:АА!
block3_conv1_18287:	А.
block3_conv2_18290:АА!
block3_conv2_18292:	А.
block3_conv3_18295:АА!
block3_conv3_18297:	А.
block4_conv1_18301:АА!
block4_conv1_18303:	А.
block4_conv2_18306:АА!
block4_conv2_18308:	А.
block4_conv3_18311:АА!
block4_conv3_18313:	А.
block5_conv1_18317:АА!
block5_conv1_18319:	А.
block5_conv2_18322:АА!
block5_conv2_18324:	А.
block5_conv3_18327:АА!
block5_conv3_18329:	А
identityИв$block1_conv1/StatefulPartitionedCallв$block1_conv2/StatefulPartitionedCallв$block2_conv1/StatefulPartitionedCallв$block2_conv2/StatefulPartitionedCallв$block3_conv1/StatefulPartitionedCallв$block3_conv2/StatefulPartitionedCallв$block3_conv3/StatefulPartitionedCallв$block4_conv1/StatefulPartitionedCallв$block4_conv2/StatefulPartitionedCallв$block4_conv3/StatefulPartitionedCallв$block5_conv1/StatefulPartitionedCallв$block5_conv2/StatefulPartitionedCallв$block5_conv3/StatefulPartitionedCallЛ
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_2block1_conv1_18263block1_conv1_18265*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЦЦ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_17540▒
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_18268block1_conv2_18270*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЦЦ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_17557я
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         KK@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_17471з
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_18274block2_conv1_18276*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         KKА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_17575░
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_18279block2_conv2_18281*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         KKА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_17592Ё
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         %%А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_17483з
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_18285block3_conv1_18287*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         %%А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_17610░
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_18290block3_conv2_18292*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         %%А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block3_conv2_layer_call_and_return_conditional_losses_17627░
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_18295block3_conv3_18297*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         %%А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block3_conv3_layer_call_and_return_conditional_losses_17644Ё
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block3_pool_layer_call_and_return_conditional_losses_17495з
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_18301block4_conv1_18303*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block4_conv1_layer_call_and_return_conditional_losses_17662░
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_18306block4_conv2_18308*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block4_conv2_layer_call_and_return_conditional_losses_17679░
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_18311block4_conv3_18313*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block4_conv3_layer_call_and_return_conditional_losses_17696Ё
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block4_pool_layer_call_and_return_conditional_losses_17507з
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_18317block5_conv1_18319*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block5_conv1_layer_call_and_return_conditional_losses_17714░
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_18322block5_conv2_18324*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block5_conv2_layer_call_and_return_conditional_losses_17731░
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_18327block5_conv3_18329*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block5_conv3_layer_call_and_return_conditional_losses_17748Ё
block5_pool/PartitionedCallPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block5_pool_layer_call_and_return_conditional_losses_17519|
IdentityIdentity$block5_pool/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         А┴
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:         ЦЦ: : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall:Z V
1
_output_shapes
:         ЦЦ
!
_user_specified_name	input_2
Т
Г
G__inference_block3_conv3_layer_call_and_return_conditional_losses_17644

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%АY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         %%Аj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         %%Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         %%А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         %%А
 
_user_specified_nameinputs
╕
▒
,__inference_sequential_1_layer_call_fn_18500
vgg16_input!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@А
	unknown_4:	А%
	unknown_5:АА
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А&

unknown_11:АА

unknown_12:	А&

unknown_13:АА

unknown_14:	А&

unknown_15:АА

unknown_16:	А&

unknown_17:АА

unknown_18:	А&

unknown_19:АА

unknown_20:	А&

unknown_21:АА

unknown_22:	А&

unknown_23:АА

unknown_24:	А

unknown_25:
А@А

unknown_26:	А

unknown_27:	А

unknown_28:
identityИвStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallvgg16_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_18437o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:         ЦЦ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:         ЦЦ
%
_user_specified_namevgg16_input
О
b
F__inference_block3_pool_layer_call_and_return_conditional_losses_19895

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ь

Ї
B__inference_dense_3_layer_call_and_return_conditional_losses_18430

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
·
д
,__inference_block3_conv1_layer_call_fn_19834

inputs#
unknown:АА
	unknown_0:	А
identityИвStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         %%А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_17610x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         %%А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         %%А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         %%А
 
_user_specified_nameinputs
О
b
F__inference_block4_pool_layer_call_and_return_conditional_losses_17507

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Т
Г
G__inference_block5_conv2_layer_call_and_return_conditional_losses_20005

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		АY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         		Аj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         		Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         		А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
Т
А
G__inference_block1_conv2_layer_call_and_return_conditional_losses_19765

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ы
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЦЦ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЦЦ@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:         ЦЦ@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:         ЦЦ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ЦЦ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ЦЦ@
 
_user_specified_nameinputs
╚
Ч
'__inference_dense_2_layer_call_fn_19694

inputs
unknown:
А@А
	unknown_0:	А
identityИвStatefulPartitionedCall█
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_18413p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А@: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А@
 
_user_specified_nameinputs
·
д
,__inference_block4_conv1_layer_call_fn_19904

inputs#
unknown:АА
	unknown_0:	А
identityИвStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block4_conv1_layer_call_and_return_conditional_losses_17662x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
О
b
F__inference_block1_pool_layer_call_and_return_conditional_losses_19775

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
·
д
,__inference_block4_conv3_layer_call_fn_19944

inputs#
unknown:АА
	unknown_0:	А
identityИвStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block4_conv3_layer_call_and_return_conditional_losses_17696x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
е

Ў
B__inference_dense_2_layer_call_and_return_conditional_losses_18413

inputs2
matmul_readvariableop_resource:
А@А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А@
 
_user_specified_nameinputs
▒
G
+__inference_block5_pool_layer_call_fn_20030

inputs
identity╫
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block5_pool_layer_call_and_return_conditional_losses_17519Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Т
Г
G__inference_block4_conv1_layer_call_and_return_conditional_losses_19915

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         Аj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
Ь
─	
G__inference_sequential_1_layer_call_and_return_conditional_losses_18661

inputs%
vgg16_18596:@
vgg16_18598:@%
vgg16_18600:@@
vgg16_18602:@&
vgg16_18604:@А
vgg16_18606:	А'
vgg16_18608:АА
vgg16_18610:	А'
vgg16_18612:АА
vgg16_18614:	А'
vgg16_18616:АА
vgg16_18618:	А'
vgg16_18620:АА
vgg16_18622:	А'
vgg16_18624:АА
vgg16_18626:	А'
vgg16_18628:АА
vgg16_18630:	А'
vgg16_18632:АА
vgg16_18634:	А'
vgg16_18636:АА
vgg16_18638:	А'
vgg16_18640:АА
vgg16_18642:	А'
vgg16_18644:АА
vgg16_18646:	А!
dense_2_18650:
А@А
dense_2_18652:	А 
dense_3_18655:	А
dense_3_18657:
identityИвdense_2/StatefulPartitionedCallвdense_3/StatefulPartitionedCallвvgg16/StatefulPartitionedCall╒
vgg16/StatefulPartitionedCallStatefulPartitionedCallinputsvgg16_18596vgg16_18598vgg16_18600vgg16_18602vgg16_18604vgg16_18606vgg16_18608vgg16_18610vgg16_18612vgg16_18614vgg16_18616vgg16_18618vgg16_18620vgg16_18622vgg16_18624vgg16_18626vgg16_18628vgg16_18630vgg16_18632vgg16_18634vgg16_18636vgg16_18638vgg16_18640vgg16_18642vgg16_18644vgg16_18646*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_vgg16_layer_call_and_return_conditional_losses_18074▌
flatten_1/PartitionedCallPartitionedCall&vgg16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_18400Й
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_18650dense_2_18652*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_18413О
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_18655dense_3_18657*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_18430w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         к
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall^vgg16/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:         ЦЦ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2>
vgg16/StatefulPartitionedCallvgg16/StatefulPartitionedCall:Y U
1
_output_shapes
:         ЦЦ
 
_user_specified_nameinputs
й
м
,__inference_sequential_1_layer_call_fn_19128

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@А
	unknown_4:	А%
	unknown_5:АА
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А&

unknown_11:АА

unknown_12:	А&

unknown_13:АА

unknown_14:	А&

unknown_15:АА

unknown_16:	А&

unknown_17:АА

unknown_18:	А&

unknown_19:АА

unknown_20:	А&

unknown_21:АА

unknown_22:	А&

unknown_23:АА

unknown_24:	А

unknown_25:
А@А

unknown_26:	А

unknown_27:	А

unknown_28:
identityИвStatefulPartitionedCall▐
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_18661o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:         ЦЦ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ЦЦ
 
_user_specified_nameinputs
Т
Г
G__inference_block2_conv2_layer_call_and_return_conditional_losses_17592

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         KKА*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         KKАY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         KKАj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         KKАw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         KKА: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         KKА
 
_user_specified_nameinputs
Т
Г
G__inference_block4_conv2_layer_call_and_return_conditional_losses_19935

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         Аj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
Ь
─	
G__inference_sequential_1_layer_call_and_return_conditional_losses_18437

inputs%
vgg16_18341:@
vgg16_18343:@%
vgg16_18345:@@
vgg16_18347:@&
vgg16_18349:@А
vgg16_18351:	А'
vgg16_18353:АА
vgg16_18355:	А'
vgg16_18357:АА
vgg16_18359:	А'
vgg16_18361:АА
vgg16_18363:	А'
vgg16_18365:АА
vgg16_18367:	А'
vgg16_18369:АА
vgg16_18371:	А'
vgg16_18373:АА
vgg16_18375:	А'
vgg16_18377:АА
vgg16_18379:	А'
vgg16_18381:АА
vgg16_18383:	А'
vgg16_18385:АА
vgg16_18387:	А'
vgg16_18389:АА
vgg16_18391:	А!
dense_2_18414:
А@А
dense_2_18416:	А 
dense_3_18431:	А
dense_3_18433:
identityИвdense_2/StatefulPartitionedCallвdense_3/StatefulPartitionedCallвvgg16/StatefulPartitionedCall╒
vgg16/StatefulPartitionedCallStatefulPartitionedCallinputsvgg16_18341vgg16_18343vgg16_18345vgg16_18347vgg16_18349vgg16_18351vgg16_18353vgg16_18355vgg16_18357vgg16_18359vgg16_18361vgg16_18363vgg16_18365vgg16_18367vgg16_18369vgg16_18371vgg16_18373vgg16_18375vgg16_18377vgg16_18379vgg16_18381vgg16_18383vgg16_18385vgg16_18387vgg16_18389vgg16_18391*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_vgg16_layer_call_and_return_conditional_losses_17756▌
flatten_1/PartitionedCallPartitionedCall&vgg16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_18400Й
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_18414dense_2_18416*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_18413О
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_18431dense_3_18433*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_18430w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         к
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall^vgg16/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:         ЦЦ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2>
vgg16/StatefulPartitionedCallvgg16/StatefulPartitionedCall:Y U
1
_output_shapes
:         ЦЦ
 
_user_specified_nameinputs
О
В
G__inference_block2_conv1_layer_call_and_return_conditional_losses_19795

inputs9
conv2d_readvariableop_resource:@А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         KKА*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         KKАY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         KKАj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         KKАw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         KK@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         KK@
 
_user_specified_nameinputs
пЗ
ї
@__inference_vgg16_layer_call_and_return_conditional_losses_19574

inputsE
+block1_conv1_conv2d_readvariableop_resource:@:
,block1_conv1_biasadd_readvariableop_resource:@E
+block1_conv2_conv2d_readvariableop_resource:@@:
,block1_conv2_biasadd_readvariableop_resource:@F
+block2_conv1_conv2d_readvariableop_resource:@А;
,block2_conv1_biasadd_readvariableop_resource:	АG
+block2_conv2_conv2d_readvariableop_resource:АА;
,block2_conv2_biasadd_readvariableop_resource:	АG
+block3_conv1_conv2d_readvariableop_resource:АА;
,block3_conv1_biasadd_readvariableop_resource:	АG
+block3_conv2_conv2d_readvariableop_resource:АА;
,block3_conv2_biasadd_readvariableop_resource:	АG
+block3_conv3_conv2d_readvariableop_resource:АА;
,block3_conv3_biasadd_readvariableop_resource:	АG
+block4_conv1_conv2d_readvariableop_resource:АА;
,block4_conv1_biasadd_readvariableop_resource:	АG
+block4_conv2_conv2d_readvariableop_resource:АА;
,block4_conv2_biasadd_readvariableop_resource:	АG
+block4_conv3_conv2d_readvariableop_resource:АА;
,block4_conv3_biasadd_readvariableop_resource:	АG
+block5_conv1_conv2d_readvariableop_resource:АА;
,block5_conv1_biasadd_readvariableop_resource:	АG
+block5_conv2_conv2d_readvariableop_resource:АА;
,block5_conv2_biasadd_readvariableop_resource:	АG
+block5_conv3_conv2d_readvariableop_resource:АА;
,block5_conv3_biasadd_readvariableop_resource:	А
identityИв#block1_conv1/BiasAdd/ReadVariableOpв"block1_conv1/Conv2D/ReadVariableOpв#block1_conv2/BiasAdd/ReadVariableOpв"block1_conv2/Conv2D/ReadVariableOpв#block2_conv1/BiasAdd/ReadVariableOpв"block2_conv1/Conv2D/ReadVariableOpв#block2_conv2/BiasAdd/ReadVariableOpв"block2_conv2/Conv2D/ReadVariableOpв#block3_conv1/BiasAdd/ReadVariableOpв"block3_conv1/Conv2D/ReadVariableOpв#block3_conv2/BiasAdd/ReadVariableOpв"block3_conv2/Conv2D/ReadVariableOpв#block3_conv3/BiasAdd/ReadVariableOpв"block3_conv3/Conv2D/ReadVariableOpв#block4_conv1/BiasAdd/ReadVariableOpв"block4_conv1/Conv2D/ReadVariableOpв#block4_conv2/BiasAdd/ReadVariableOpв"block4_conv2/Conv2D/ReadVariableOpв#block4_conv3/BiasAdd/ReadVariableOpв"block4_conv3/Conv2D/ReadVariableOpв#block5_conv1/BiasAdd/ReadVariableOpв"block5_conv1/Conv2D/ReadVariableOpв#block5_conv2/BiasAdd/ReadVariableOpв"block5_conv2/Conv2D/ReadVariableOpв#block5_conv3/BiasAdd/ReadVariableOpв"block5_conv3/Conv2D/ReadVariableOpЦ
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0╡
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЦЦ@*
paddingSAME*
strides
М
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ж
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЦЦ@t
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:         ЦЦ@Ц
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0╬
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЦЦ@*
paddingSAME*
strides
М
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ж
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЦЦ@t
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:         ЦЦ@м
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*/
_output_shapes
:         KK@*
ksize
*
paddingVALID*
strides
Ч
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0╩
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         KKА*
paddingSAME*
strides
Н
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0е
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         KKАs
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:         KKАШ
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0═
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         KKА*
paddingSAME*
strides
Н
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0е
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         KKАs
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:         KKАн
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:         %%А*
ksize
*
paddingVALID*
strides
Ш
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0╩
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
Н
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0е
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%Аs
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:         %%АШ
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0═
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
Н
#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0е
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%Аs
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:         %%АШ
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0═
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
Н
#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0е
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%Аs
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:         %%Ан
block3_pool/MaxPoolMaxPoolblock3_conv3/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
Ш
"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0╩
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
Н
#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0е
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Аs
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:         АШ
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0═
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
Н
#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0е
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Аs
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:         АШ
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0═
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
Н
#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0е
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Аs
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:         Ан
block4_pool/MaxPoolMaxPoolblock4_conv3/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
Ш
"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0╩
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
Н
#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0е
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		Аs
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:         		АШ
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0═
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
Н
#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0е
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		Аs
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:         		АШ
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp+block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0═
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
Н
#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp,block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0е
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		Аs
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:         		Ан
block5_pool/MaxPoolMaxPoolblock5_conv3/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
t
IdentityIdentityblock5_pool/MaxPool:output:0^NoOp*
T0*0
_output_shapes
:         АХ
NoOpNoOp$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp$^block5_conv2/BiasAdd/ReadVariableOp#^block5_conv2/Conv2D/ReadVariableOp$^block5_conv3/BiasAdd/ReadVariableOp#^block5_conv3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:         ЦЦ: : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp2J
#block4_conv2/BiasAdd/ReadVariableOp#block4_conv2/BiasAdd/ReadVariableOp2H
"block4_conv2/Conv2D/ReadVariableOp"block4_conv2/Conv2D/ReadVariableOp2J
#block4_conv3/BiasAdd/ReadVariableOp#block4_conv3/BiasAdd/ReadVariableOp2H
"block4_conv3/Conv2D/ReadVariableOp"block4_conv3/Conv2D/ReadVariableOp2J
#block5_conv1/BiasAdd/ReadVariableOp#block5_conv1/BiasAdd/ReadVariableOp2H
"block5_conv1/Conv2D/ReadVariableOp"block5_conv1/Conv2D/ReadVariableOp2J
#block5_conv2/BiasAdd/ReadVariableOp#block5_conv2/BiasAdd/ReadVariableOp2H
"block5_conv2/Conv2D/ReadVariableOp"block5_conv2/Conv2D/ReadVariableOp2J
#block5_conv3/BiasAdd/ReadVariableOp#block5_conv3/BiasAdd/ReadVariableOp2H
"block5_conv3/Conv2D/ReadVariableOp"block5_conv3/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:         ЦЦ
 
_user_specified_nameinputs
╚
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_19685

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"        ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         А@Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
√
б
,__inference_block1_conv2_layer_call_fn_19754

inputs!
unknown:@@
	unknown_0:@
identityИвStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЦЦ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_17557y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЦЦ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ЦЦ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ЦЦ@
 
_user_specified_nameinputs
─
Х
'__inference_dense_3_layer_call_fn_19714

inputs
unknown:	А
	unknown_0:
identityИвStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_18430o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╚
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_18400

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"        ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         А@Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         А@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         А:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
Т
Г
G__inference_block4_conv3_layer_call_and_return_conditional_losses_19955

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         Аj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
Т
Г
G__inference_block4_conv2_layer_call_and_return_conditional_losses_17679

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         Аj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
ў
г
,__inference_block2_conv1_layer_call_fn_19784

inputs"
unknown:@А
	unknown_0:	А
identityИвStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         KKА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_17575x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         KKА`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         KK@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         KK@
 
_user_specified_nameinputs
·
д
,__inference_block2_conv2_layer_call_fn_19804

inputs#
unknown:АА
	unknown_0:	А
identityИвStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         KKА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_17592x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         KKА`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         KKА: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         KKА
 
_user_specified_nameinputs
·
д
,__inference_block5_conv2_layer_call_fn_19994

inputs#
unknown:АА
	unknown_0:	А
identityИвStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block5_conv2_layer_call_and_return_conditional_losses_17731x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         		А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         		А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
Т
Г
G__inference_block3_conv2_layer_call_and_return_conditional_losses_17627

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%АY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         %%Аj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         %%Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         %%А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         %%А
 
_user_specified_nameinputs
∙
▒
%__inference_vgg16_layer_call_fn_19474

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@А
	unknown_4:	А%
	unknown_5:АА
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А&

unknown_11:АА

unknown_12:	А&

unknown_13:АА

unknown_14:	А&

unknown_15:АА

unknown_16:	А&

unknown_17:АА

unknown_18:	А&

unknown_19:АА

unknown_20:	А&

unknown_21:АА

unknown_22:	А&

unknown_23:АА

unknown_24:	А
identityИвStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_vgg16_layer_call_and_return_conditional_losses_18074x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:         ЦЦ: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ЦЦ
 
_user_specified_nameinputs
Т
Г
G__inference_block3_conv3_layer_call_and_return_conditional_losses_19885

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%АY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         %%Аj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         %%Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         %%А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         %%А
 
_user_specified_nameinputs
╕
▒
,__inference_sequential_1_layer_call_fn_18789
vgg16_input!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@А
	unknown_4:	А%
	unknown_5:АА
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А&

unknown_11:АА

unknown_12:	А&

unknown_13:АА

unknown_14:	А&

unknown_15:АА

unknown_16:	А&

unknown_17:АА

unknown_18:	А&

unknown_19:АА

unknown_20:	А&

unknown_21:АА

unknown_22:	А&

unknown_23:АА

unknown_24:	А

unknown_25:
А@А

unknown_26:	А

unknown_27:	А

unknown_28:
identityИвStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallvgg16_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_18661o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:         ЦЦ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:         ЦЦ
%
_user_specified_namevgg16_input
О
b
F__inference_block5_pool_layer_call_and_return_conditional_losses_17519

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Т
Г
G__inference_block5_conv1_layer_call_and_return_conditional_losses_17714

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		АY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         		Аj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         		Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         		А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
▐V
Л
@__inference_vgg16_layer_call_and_return_conditional_losses_18260
input_2,
block1_conv1_18189:@ 
block1_conv1_18191:@,
block1_conv2_18194:@@ 
block1_conv2_18196:@-
block2_conv1_18200:@А!
block2_conv1_18202:	А.
block2_conv2_18205:АА!
block2_conv2_18207:	А.
block3_conv1_18211:АА!
block3_conv1_18213:	А.
block3_conv2_18216:АА!
block3_conv2_18218:	А.
block3_conv3_18221:АА!
block3_conv3_18223:	А.
block4_conv1_18227:АА!
block4_conv1_18229:	А.
block4_conv2_18232:АА!
block4_conv2_18234:	А.
block4_conv3_18237:АА!
block4_conv3_18239:	А.
block5_conv1_18243:АА!
block5_conv1_18245:	А.
block5_conv2_18248:АА!
block5_conv2_18250:	А.
block5_conv3_18253:АА!
block5_conv3_18255:	А
identityИв$block1_conv1/StatefulPartitionedCallв$block1_conv2/StatefulPartitionedCallв$block2_conv1/StatefulPartitionedCallв$block2_conv2/StatefulPartitionedCallв$block3_conv1/StatefulPartitionedCallв$block3_conv2/StatefulPartitionedCallв$block3_conv3/StatefulPartitionedCallв$block4_conv1/StatefulPartitionedCallв$block4_conv2/StatefulPartitionedCallв$block4_conv3/StatefulPartitionedCallв$block5_conv1/StatefulPartitionedCallв$block5_conv2/StatefulPartitionedCallв$block5_conv3/StatefulPartitionedCallЛ
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_2block1_conv1_18189block1_conv1_18191*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЦЦ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_17540▒
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_18194block1_conv2_18196*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЦЦ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_17557я
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         KK@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_17471з
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_18200block2_conv1_18202*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         KKА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_17575░
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_18205block2_conv2_18207*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         KKА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_17592Ё
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         %%А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_17483з
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_18211block3_conv1_18213*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         %%А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_17610░
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_18216block3_conv2_18218*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         %%А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block3_conv2_layer_call_and_return_conditional_losses_17627░
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_18221block3_conv3_18223*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         %%А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block3_conv3_layer_call_and_return_conditional_losses_17644Ё
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block3_pool_layer_call_and_return_conditional_losses_17495з
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_18227block4_conv1_18229*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block4_conv1_layer_call_and_return_conditional_losses_17662░
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_18232block4_conv2_18234*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block4_conv2_layer_call_and_return_conditional_losses_17679░
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_18237block4_conv3_18239*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block4_conv3_layer_call_and_return_conditional_losses_17696Ё
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block4_pool_layer_call_and_return_conditional_losses_17507з
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_18243block5_conv1_18245*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block5_conv1_layer_call_and_return_conditional_losses_17714░
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_18248block5_conv2_18250*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block5_conv2_layer_call_and_return_conditional_losses_17731░
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_18253block5_conv3_18255*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block5_conv3_layer_call_and_return_conditional_losses_17748Ё
block5_pool/PartitionedCallPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block5_pool_layer_call_and_return_conditional_losses_17519|
IdentityIdentity$block5_pool/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         А┴
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:         ЦЦ: : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall:Z V
1
_output_shapes
:         ЦЦ
!
_user_specified_name	input_2
█V
К
@__inference_vgg16_layer_call_and_return_conditional_losses_17756

inputs,
block1_conv1_17541:@ 
block1_conv1_17543:@,
block1_conv2_17558:@@ 
block1_conv2_17560:@-
block2_conv1_17576:@А!
block2_conv1_17578:	А.
block2_conv2_17593:АА!
block2_conv2_17595:	А.
block3_conv1_17611:АА!
block3_conv1_17613:	А.
block3_conv2_17628:АА!
block3_conv2_17630:	А.
block3_conv3_17645:АА!
block3_conv3_17647:	А.
block4_conv1_17663:АА!
block4_conv1_17665:	А.
block4_conv2_17680:АА!
block4_conv2_17682:	А.
block4_conv3_17697:АА!
block4_conv3_17699:	А.
block5_conv1_17715:АА!
block5_conv1_17717:	А.
block5_conv2_17732:АА!
block5_conv2_17734:	А.
block5_conv3_17749:АА!
block5_conv3_17751:	А
identityИв$block1_conv1/StatefulPartitionedCallв$block1_conv2/StatefulPartitionedCallв$block2_conv1/StatefulPartitionedCallв$block2_conv2/StatefulPartitionedCallв$block3_conv1/StatefulPartitionedCallв$block3_conv2/StatefulPartitionedCallв$block3_conv3/StatefulPartitionedCallв$block4_conv1/StatefulPartitionedCallв$block4_conv2/StatefulPartitionedCallв$block4_conv3/StatefulPartitionedCallв$block5_conv1/StatefulPartitionedCallв$block5_conv2/StatefulPartitionedCallв$block5_conv3/StatefulPartitionedCallК
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv1_17541block1_conv1_17543*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЦЦ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_17540▒
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_17558block1_conv2_17560*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЦЦ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_17557я
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         KK@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_17471з
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_17576block2_conv1_17578*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         KKА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_17575░
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_17593block2_conv2_17595*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         KKА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_17592Ё
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         %%А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_17483з
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_17611block3_conv1_17613*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         %%А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_17610░
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_17628block3_conv2_17630*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         %%А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block3_conv2_layer_call_and_return_conditional_losses_17627░
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_17645block3_conv3_17647*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         %%А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block3_conv3_layer_call_and_return_conditional_losses_17644Ё
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block3_pool_layer_call_and_return_conditional_losses_17495з
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_17663block4_conv1_17665*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block4_conv1_layer_call_and_return_conditional_losses_17662░
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_17680block4_conv2_17682*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block4_conv2_layer_call_and_return_conditional_losses_17679░
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_17697block4_conv3_17699*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block4_conv3_layer_call_and_return_conditional_losses_17696Ё
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block4_pool_layer_call_and_return_conditional_losses_17507з
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_17715block5_conv1_17717*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block5_conv1_layer_call_and_return_conditional_losses_17714░
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_17732block5_conv2_17734*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block5_conv2_layer_call_and_return_conditional_losses_17731░
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_17749block5_conv3_17751*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block5_conv3_layer_call_and_return_conditional_losses_17748Ё
block5_pool/PartitionedCallPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block5_pool_layer_call_and_return_conditional_losses_17519|
IdentityIdentity$block5_pool/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         А┴
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:         ЦЦ: : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall:Y U
1
_output_shapes
:         ЦЦ
 
_user_specified_nameinputs
█V
К
@__inference_vgg16_layer_call_and_return_conditional_losses_18074

inputs,
block1_conv1_18003:@ 
block1_conv1_18005:@,
block1_conv2_18008:@@ 
block1_conv2_18010:@-
block2_conv1_18014:@А!
block2_conv1_18016:	А.
block2_conv2_18019:АА!
block2_conv2_18021:	А.
block3_conv1_18025:АА!
block3_conv1_18027:	А.
block3_conv2_18030:АА!
block3_conv2_18032:	А.
block3_conv3_18035:АА!
block3_conv3_18037:	А.
block4_conv1_18041:АА!
block4_conv1_18043:	А.
block4_conv2_18046:АА!
block4_conv2_18048:	А.
block4_conv3_18051:АА!
block4_conv3_18053:	А.
block5_conv1_18057:АА!
block5_conv1_18059:	А.
block5_conv2_18062:АА!
block5_conv2_18064:	А.
block5_conv3_18067:АА!
block5_conv3_18069:	А
identityИв$block1_conv1/StatefulPartitionedCallв$block1_conv2/StatefulPartitionedCallв$block2_conv1/StatefulPartitionedCallв$block2_conv2/StatefulPartitionedCallв$block3_conv1/StatefulPartitionedCallв$block3_conv2/StatefulPartitionedCallв$block3_conv3/StatefulPartitionedCallв$block4_conv1/StatefulPartitionedCallв$block4_conv2/StatefulPartitionedCallв$block4_conv3/StatefulPartitionedCallв$block5_conv1/StatefulPartitionedCallв$block5_conv2/StatefulPartitionedCallв$block5_conv3/StatefulPartitionedCallК
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv1_18003block1_conv1_18005*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЦЦ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_17540▒
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_18008block1_conv2_18010*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЦЦ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block1_conv2_layer_call_and_return_conditional_losses_17557я
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         KK@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_17471з
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_18014block2_conv1_18016*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         KKА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block2_conv1_layer_call_and_return_conditional_losses_17575░
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_18019block2_conv2_18021*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         KKА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block2_conv2_layer_call_and_return_conditional_losses_17592Ё
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         %%А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_17483з
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_18025block3_conv1_18027*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         %%А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block3_conv1_layer_call_and_return_conditional_losses_17610░
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_18030block3_conv2_18032*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         %%А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block3_conv2_layer_call_and_return_conditional_losses_17627░
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_18035block3_conv3_18037*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         %%А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block3_conv3_layer_call_and_return_conditional_losses_17644Ё
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block3_pool_layer_call_and_return_conditional_losses_17495з
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_18041block4_conv1_18043*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block4_conv1_layer_call_and_return_conditional_losses_17662░
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_18046block4_conv2_18048*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block4_conv2_layer_call_and_return_conditional_losses_17679░
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_18051block4_conv3_18053*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block4_conv3_layer_call_and_return_conditional_losses_17696Ё
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block4_pool_layer_call_and_return_conditional_losses_17507з
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_18057block5_conv1_18059*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block5_conv1_layer_call_and_return_conditional_losses_17714░
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_18062block5_conv2_18064*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block5_conv2_layer_call_and_return_conditional_losses_17731░
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_18067block5_conv3_18069*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block5_conv3_layer_call_and_return_conditional_losses_17748Ё
block5_pool/PartitionedCallPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block5_pool_layer_call_and_return_conditional_losses_17519|
IdentityIdentity$block5_pool/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         А┴
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:         ЦЦ: : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall:Y U
1
_output_shapes
:         ЦЦ
 
_user_specified_nameinputs
∙
▒
%__inference_vgg16_layer_call_fn_19417

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@А
	unknown_4:	А%
	unknown_5:АА
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А&

unknown_11:АА

unknown_12:	А&

unknown_13:АА

unknown_14:	А&

unknown_15:АА

unknown_16:	А&

unknown_17:АА

unknown_18:	А&

unknown_19:АА

unknown_20:	А&

unknown_21:АА

unknown_22:	А&

unknown_23:АА

unknown_24:	А
identityИвStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_vgg16_layer_call_and_return_conditional_losses_17756x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:         ЦЦ: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ЦЦ
 
_user_specified_nameinputs
О
b
F__inference_block5_pool_layer_call_and_return_conditional_losses_20035

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ЫY
ї
__inference__traced_save_20199
file_prefix-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop2
.savev2_block1_conv1_kernel_read_readvariableop0
,savev2_block1_conv1_bias_read_readvariableop2
.savev2_block1_conv2_kernel_read_readvariableop0
,savev2_block1_conv2_bias_read_readvariableop2
.savev2_block2_conv1_kernel_read_readvariableop0
,savev2_block2_conv1_bias_read_readvariableop2
.savev2_block2_conv2_kernel_read_readvariableop0
,savev2_block2_conv2_bias_read_readvariableop2
.savev2_block3_conv1_kernel_read_readvariableop0
,savev2_block3_conv1_bias_read_readvariableop2
.savev2_block3_conv2_kernel_read_readvariableop0
,savev2_block3_conv2_bias_read_readvariableop2
.savev2_block3_conv3_kernel_read_readvariableop0
,savev2_block3_conv3_bias_read_readvariableop2
.savev2_block4_conv1_kernel_read_readvariableop0
,savev2_block4_conv1_bias_read_readvariableop2
.savev2_block4_conv2_kernel_read_readvariableop0
,savev2_block4_conv2_bias_read_readvariableop2
.savev2_block4_conv3_kernel_read_readvariableop0
,savev2_block4_conv3_bias_read_readvariableop2
.savev2_block5_conv1_kernel_read_readvariableop0
,savev2_block5_conv1_bias_read_readvariableop2
.savev2_block5_conv2_kernel_read_readvariableop0
,savev2_block5_conv2_bias_read_readvariableop2
.savev2_block5_conv3_kernel_read_readvariableop0
,savev2_block5_conv3_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableop
savev2_const

identity_1ИвMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: █
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*Д
value·Bў0B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH═
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B н
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop.savev2_block1_conv1_kernel_read_readvariableop,savev2_block1_conv1_bias_read_readvariableop.savev2_block1_conv2_kernel_read_readvariableop,savev2_block1_conv2_bias_read_readvariableop.savev2_block2_conv1_kernel_read_readvariableop,savev2_block2_conv1_bias_read_readvariableop.savev2_block2_conv2_kernel_read_readvariableop,savev2_block2_conv2_bias_read_readvariableop.savev2_block3_conv1_kernel_read_readvariableop,savev2_block3_conv1_bias_read_readvariableop.savev2_block3_conv2_kernel_read_readvariableop,savev2_block3_conv2_bias_read_readvariableop.savev2_block3_conv3_kernel_read_readvariableop,savev2_block3_conv3_bias_read_readvariableop.savev2_block4_conv1_kernel_read_readvariableop,savev2_block4_conv1_bias_read_readvariableop.savev2_block4_conv2_kernel_read_readvariableop,savev2_block4_conv2_bias_read_readvariableop.savev2_block4_conv3_kernel_read_readvariableop,savev2_block4_conv3_bias_read_readvariableop.savev2_block5_conv1_kernel_read_readvariableop,savev2_block5_conv1_bias_read_readvariableop.savev2_block5_conv2_kernel_read_readvariableop,savev2_block5_conv2_bias_read_readvariableop.savev2_block5_conv3_kernel_read_readvariableop,savev2_block5_conv3_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *>
dtypes4
220	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*я
_input_shapes▌
┌: :
А@А:А:	А::@:@:@@:@:@А:А:АА:А:АА:А:АА:А:АА:А:АА:А:АА:А:АА:А:АА:А:АА:А:АА:А: : : : : : : : : :
А@А:А:	А::
А@А:А:	А:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
А@А:!

_output_shapes	
:А:%!

_output_shapes
:	А: 

_output_shapes
::,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:-	)
'
_output_shapes
:@А:!


_output_shapes	
:А:.*
(
_output_shapes
:АА:!

_output_shapes	
:А:.*
(
_output_shapes
:АА:!

_output_shapes	
:А:.*
(
_output_shapes
:АА:!

_output_shapes	
:А:.*
(
_output_shapes
:АА:!

_output_shapes	
:А:.*
(
_output_shapes
:АА:!

_output_shapes	
:А:.*
(
_output_shapes
:АА:!

_output_shapes	
:А:.*
(
_output_shapes
:АА:!

_output_shapes	
:А:.*
(
_output_shapes
:АА:!

_output_shapes	
:А:.*
(
_output_shapes
:АА:!

_output_shapes	
:А:.*
(
_output_shapes
:АА:!

_output_shapes	
:А:

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :&("
 
_output_shapes
:
А@А:!)

_output_shapes	
:А:%*!

_output_shapes
:	А: +

_output_shapes
::&,"
 
_output_shapes
:
А@А:!-

_output_shapes	
:А:%.!

_output_shapes
:	А: /

_output_shapes
::0

_output_shapes
: 
О
b
F__inference_block4_pool_layer_call_and_return_conditional_losses_19965

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Т
Г
G__inference_block4_conv1_layer_call_and_return_conditional_losses_17662

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         Аj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
Т
Г
G__inference_block5_conv3_layer_call_and_return_conditional_losses_17748

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		АY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         		Аj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         		Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         		А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
гз
Ь
G__inference_sequential_1_layer_call_and_return_conditional_losses_19244

inputsK
1vgg16_block1_conv1_conv2d_readvariableop_resource:@@
2vgg16_block1_conv1_biasadd_readvariableop_resource:@K
1vgg16_block1_conv2_conv2d_readvariableop_resource:@@@
2vgg16_block1_conv2_biasadd_readvariableop_resource:@L
1vgg16_block2_conv1_conv2d_readvariableop_resource:@АA
2vgg16_block2_conv1_biasadd_readvariableop_resource:	АM
1vgg16_block2_conv2_conv2d_readvariableop_resource:ААA
2vgg16_block2_conv2_biasadd_readvariableop_resource:	АM
1vgg16_block3_conv1_conv2d_readvariableop_resource:ААA
2vgg16_block3_conv1_biasadd_readvariableop_resource:	АM
1vgg16_block3_conv2_conv2d_readvariableop_resource:ААA
2vgg16_block3_conv2_biasadd_readvariableop_resource:	АM
1vgg16_block3_conv3_conv2d_readvariableop_resource:ААA
2vgg16_block3_conv3_biasadd_readvariableop_resource:	АM
1vgg16_block4_conv1_conv2d_readvariableop_resource:ААA
2vgg16_block4_conv1_biasadd_readvariableop_resource:	АM
1vgg16_block4_conv2_conv2d_readvariableop_resource:ААA
2vgg16_block4_conv2_biasadd_readvariableop_resource:	АM
1vgg16_block4_conv3_conv2d_readvariableop_resource:ААA
2vgg16_block4_conv3_biasadd_readvariableop_resource:	АM
1vgg16_block5_conv1_conv2d_readvariableop_resource:ААA
2vgg16_block5_conv1_biasadd_readvariableop_resource:	АM
1vgg16_block5_conv2_conv2d_readvariableop_resource:ААA
2vgg16_block5_conv2_biasadd_readvariableop_resource:	АM
1vgg16_block5_conv3_conv2d_readvariableop_resource:ААA
2vgg16_block5_conv3_biasadd_readvariableop_resource:	А:
&dense_2_matmul_readvariableop_resource:
А@А6
'dense_2_biasadd_readvariableop_resource:	А9
&dense_3_matmul_readvariableop_resource:	А5
'dense_3_biasadd_readvariableop_resource:
identityИвdense_2/BiasAdd/ReadVariableOpвdense_2/MatMul/ReadVariableOpвdense_3/BiasAdd/ReadVariableOpвdense_3/MatMul/ReadVariableOpв)vgg16/block1_conv1/BiasAdd/ReadVariableOpв(vgg16/block1_conv1/Conv2D/ReadVariableOpв)vgg16/block1_conv2/BiasAdd/ReadVariableOpв(vgg16/block1_conv2/Conv2D/ReadVariableOpв)vgg16/block2_conv1/BiasAdd/ReadVariableOpв(vgg16/block2_conv1/Conv2D/ReadVariableOpв)vgg16/block2_conv2/BiasAdd/ReadVariableOpв(vgg16/block2_conv2/Conv2D/ReadVariableOpв)vgg16/block3_conv1/BiasAdd/ReadVariableOpв(vgg16/block3_conv1/Conv2D/ReadVariableOpв)vgg16/block3_conv2/BiasAdd/ReadVariableOpв(vgg16/block3_conv2/Conv2D/ReadVariableOpв)vgg16/block3_conv3/BiasAdd/ReadVariableOpв(vgg16/block3_conv3/Conv2D/ReadVariableOpв)vgg16/block4_conv1/BiasAdd/ReadVariableOpв(vgg16/block4_conv1/Conv2D/ReadVariableOpв)vgg16/block4_conv2/BiasAdd/ReadVariableOpв(vgg16/block4_conv2/Conv2D/ReadVariableOpв)vgg16/block4_conv3/BiasAdd/ReadVariableOpв(vgg16/block4_conv3/Conv2D/ReadVariableOpв)vgg16/block5_conv1/BiasAdd/ReadVariableOpв(vgg16/block5_conv1/Conv2D/ReadVariableOpв)vgg16/block5_conv2/BiasAdd/ReadVariableOpв(vgg16/block5_conv2/Conv2D/ReadVariableOpв)vgg16/block5_conv3/BiasAdd/ReadVariableOpв(vgg16/block5_conv3/Conv2D/ReadVariableOpв
(vgg16/block1_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0┴
vgg16/block1_conv1/Conv2DConv2Dinputs0vgg16/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЦЦ@*
paddingSAME*
strides
Ш
)vgg16/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╕
vgg16/block1_conv1/BiasAddBiasAdd"vgg16/block1_conv1/Conv2D:output:01vgg16/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЦЦ@А
vgg16/block1_conv1/ReluRelu#vgg16/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:         ЦЦ@в
(vgg16/block1_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0р
vgg16/block1_conv2/Conv2DConv2D%vgg16/block1_conv1/Relu:activations:00vgg16/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЦЦ@*
paddingSAME*
strides
Ш
)vgg16/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╕
vgg16/block1_conv2/BiasAddBiasAdd"vgg16/block1_conv2/Conv2D:output:01vgg16/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЦЦ@А
vgg16/block1_conv2/ReluRelu#vgg16/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:         ЦЦ@╕
vgg16/block1_pool/MaxPoolMaxPool%vgg16/block1_conv2/Relu:activations:0*/
_output_shapes
:         KK@*
ksize
*
paddingVALID*
strides
г
(vgg16/block2_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0▄
vgg16/block2_conv1/Conv2DConv2D"vgg16/block1_pool/MaxPool:output:00vgg16/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         KKА*
paddingSAME*
strides
Щ
)vgg16/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
vgg16/block2_conv1/BiasAddBiasAdd"vgg16/block2_conv1/Conv2D:output:01vgg16/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         KKА
vgg16/block2_conv1/ReluRelu#vgg16/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:         KKАд
(vgg16/block2_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0▀
vgg16/block2_conv2/Conv2DConv2D%vgg16/block2_conv1/Relu:activations:00vgg16/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         KKА*
paddingSAME*
strides
Щ
)vgg16/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
vgg16/block2_conv2/BiasAddBiasAdd"vgg16/block2_conv2/Conv2D:output:01vgg16/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         KKА
vgg16/block2_conv2/ReluRelu#vgg16/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:         KKА╣
vgg16/block2_pool/MaxPoolMaxPool%vgg16/block2_conv2/Relu:activations:0*0
_output_shapes
:         %%А*
ksize
*
paddingVALID*
strides
д
(vgg16/block3_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0▄
vgg16/block3_conv1/Conv2DConv2D"vgg16/block2_pool/MaxPool:output:00vgg16/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
Щ
)vgg16/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
vgg16/block3_conv1/BiasAddBiasAdd"vgg16/block3_conv1/Conv2D:output:01vgg16/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А
vgg16/block3_conv1/ReluRelu#vgg16/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:         %%Ад
(vgg16/block3_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0▀
vgg16/block3_conv2/Conv2DConv2D%vgg16/block3_conv1/Relu:activations:00vgg16/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
Щ
)vgg16/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
vgg16/block3_conv2/BiasAddBiasAdd"vgg16/block3_conv2/Conv2D:output:01vgg16/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А
vgg16/block3_conv2/ReluRelu#vgg16/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:         %%Ад
(vgg16/block3_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg16_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0▀
vgg16/block3_conv3/Conv2DConv2D%vgg16/block3_conv2/Relu:activations:00vgg16/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
Щ
)vgg16/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
vgg16/block3_conv3/BiasAddBiasAdd"vgg16/block3_conv3/Conv2D:output:01vgg16/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А
vgg16/block3_conv3/ReluRelu#vgg16/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А╣
vgg16/block3_pool/MaxPoolMaxPool%vgg16/block3_conv3/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
д
(vgg16/block4_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0▄
vgg16/block4_conv1/Conv2DConv2D"vgg16/block3_pool/MaxPool:output:00vgg16/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
Щ
)vgg16/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
vgg16/block4_conv1/BiasAddBiasAdd"vgg16/block4_conv1/Conv2D:output:01vgg16/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А
vgg16/block4_conv1/ReluRelu#vgg16/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:         Ад
(vgg16/block4_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0▀
vgg16/block4_conv2/Conv2DConv2D%vgg16/block4_conv1/Relu:activations:00vgg16/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
Щ
)vgg16/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
vgg16/block4_conv2/BiasAddBiasAdd"vgg16/block4_conv2/Conv2D:output:01vgg16/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А
vgg16/block4_conv2/ReluRelu#vgg16/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:         Ад
(vgg16/block4_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg16_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0▀
vgg16/block4_conv3/Conv2DConv2D%vgg16/block4_conv2/Relu:activations:00vgg16/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
Щ
)vgg16/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
vgg16/block4_conv3/BiasAddBiasAdd"vgg16/block4_conv3/Conv2D:output:01vgg16/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А
vgg16/block4_conv3/ReluRelu#vgg16/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:         А╣
vgg16/block4_pool/MaxPoolMaxPool%vgg16/block4_conv3/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
д
(vgg16/block5_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0▄
vgg16/block5_conv1/Conv2DConv2D"vgg16/block4_pool/MaxPool:output:00vgg16/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
Щ
)vgg16/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
vgg16/block5_conv1/BiasAddBiasAdd"vgg16/block5_conv1/Conv2D:output:01vgg16/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А
vgg16/block5_conv1/ReluRelu#vgg16/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:         		Ад
(vgg16/block5_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0▀
vgg16/block5_conv2/Conv2DConv2D%vgg16/block5_conv1/Relu:activations:00vgg16/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
Щ
)vgg16/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
vgg16/block5_conv2/BiasAddBiasAdd"vgg16/block5_conv2/Conv2D:output:01vgg16/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А
vgg16/block5_conv2/ReluRelu#vgg16/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:         		Ад
(vgg16/block5_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg16_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0▀
vgg16/block5_conv3/Conv2DConv2D%vgg16/block5_conv2/Relu:activations:00vgg16/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
Щ
)vgg16/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
vgg16/block5_conv3/BiasAddBiasAdd"vgg16/block5_conv3/Conv2D:output:01vgg16/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А
vgg16/block5_conv3/ReluRelu#vgg16/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:         		А╣
vgg16/block5_pool/MaxPoolMaxPool%vgg16/block5_conv3/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"        Н
flatten_1/ReshapeReshape"vgg16/block5_pool/MaxPool:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:         А@Ж
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype0О
dense_2/MatMulMatMulflatten_1/Reshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АГ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0П
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аa
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:         АЕ
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Н
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         В
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         f
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:         b
IdentityIdentitydense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:         │

NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*^vgg16/block1_conv1/BiasAdd/ReadVariableOp)^vgg16/block1_conv1/Conv2D/ReadVariableOp*^vgg16/block1_conv2/BiasAdd/ReadVariableOp)^vgg16/block1_conv2/Conv2D/ReadVariableOp*^vgg16/block2_conv1/BiasAdd/ReadVariableOp)^vgg16/block2_conv1/Conv2D/ReadVariableOp*^vgg16/block2_conv2/BiasAdd/ReadVariableOp)^vgg16/block2_conv2/Conv2D/ReadVariableOp*^vgg16/block3_conv1/BiasAdd/ReadVariableOp)^vgg16/block3_conv1/Conv2D/ReadVariableOp*^vgg16/block3_conv2/BiasAdd/ReadVariableOp)^vgg16/block3_conv2/Conv2D/ReadVariableOp*^vgg16/block3_conv3/BiasAdd/ReadVariableOp)^vgg16/block3_conv3/Conv2D/ReadVariableOp*^vgg16/block4_conv1/BiasAdd/ReadVariableOp)^vgg16/block4_conv1/Conv2D/ReadVariableOp*^vgg16/block4_conv2/BiasAdd/ReadVariableOp)^vgg16/block4_conv2/Conv2D/ReadVariableOp*^vgg16/block4_conv3/BiasAdd/ReadVariableOp)^vgg16/block4_conv3/Conv2D/ReadVariableOp*^vgg16/block5_conv1/BiasAdd/ReadVariableOp)^vgg16/block5_conv1/Conv2D/ReadVariableOp*^vgg16/block5_conv2/BiasAdd/ReadVariableOp)^vgg16/block5_conv2/Conv2D/ReadVariableOp*^vgg16/block5_conv3/BiasAdd/ReadVariableOp)^vgg16/block5_conv3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:         ЦЦ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2V
)vgg16/block1_conv1/BiasAdd/ReadVariableOp)vgg16/block1_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block1_conv1/Conv2D/ReadVariableOp(vgg16/block1_conv1/Conv2D/ReadVariableOp2V
)vgg16/block1_conv2/BiasAdd/ReadVariableOp)vgg16/block1_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block1_conv2/Conv2D/ReadVariableOp(vgg16/block1_conv2/Conv2D/ReadVariableOp2V
)vgg16/block2_conv1/BiasAdd/ReadVariableOp)vgg16/block2_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block2_conv1/Conv2D/ReadVariableOp(vgg16/block2_conv1/Conv2D/ReadVariableOp2V
)vgg16/block2_conv2/BiasAdd/ReadVariableOp)vgg16/block2_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block2_conv2/Conv2D/ReadVariableOp(vgg16/block2_conv2/Conv2D/ReadVariableOp2V
)vgg16/block3_conv1/BiasAdd/ReadVariableOp)vgg16/block3_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block3_conv1/Conv2D/ReadVariableOp(vgg16/block3_conv1/Conv2D/ReadVariableOp2V
)vgg16/block3_conv2/BiasAdd/ReadVariableOp)vgg16/block3_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block3_conv2/Conv2D/ReadVariableOp(vgg16/block3_conv2/Conv2D/ReadVariableOp2V
)vgg16/block3_conv3/BiasAdd/ReadVariableOp)vgg16/block3_conv3/BiasAdd/ReadVariableOp2T
(vgg16/block3_conv3/Conv2D/ReadVariableOp(vgg16/block3_conv3/Conv2D/ReadVariableOp2V
)vgg16/block4_conv1/BiasAdd/ReadVariableOp)vgg16/block4_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block4_conv1/Conv2D/ReadVariableOp(vgg16/block4_conv1/Conv2D/ReadVariableOp2V
)vgg16/block4_conv2/BiasAdd/ReadVariableOp)vgg16/block4_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block4_conv2/Conv2D/ReadVariableOp(vgg16/block4_conv2/Conv2D/ReadVariableOp2V
)vgg16/block4_conv3/BiasAdd/ReadVariableOp)vgg16/block4_conv3/BiasAdd/ReadVariableOp2T
(vgg16/block4_conv3/Conv2D/ReadVariableOp(vgg16/block4_conv3/Conv2D/ReadVariableOp2V
)vgg16/block5_conv1/BiasAdd/ReadVariableOp)vgg16/block5_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block5_conv1/Conv2D/ReadVariableOp(vgg16/block5_conv1/Conv2D/ReadVariableOp2V
)vgg16/block5_conv2/BiasAdd/ReadVariableOp)vgg16/block5_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block5_conv2/Conv2D/ReadVariableOp(vgg16/block5_conv2/Conv2D/ReadVariableOp2V
)vgg16/block5_conv3/BiasAdd/ReadVariableOp)vgg16/block5_conv3/BiasAdd/ReadVariableOp2T
(vgg16/block5_conv3/Conv2D/ReadVariableOp(vgg16/block5_conv3/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:         ЦЦ
 
_user_specified_nameinputs
·
д
,__inference_block5_conv1_layer_call_fn_19974

inputs#
unknown:АА
	unknown_0:	А
identityИвStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block5_conv1_layer_call_and_return_conditional_losses_17714x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         		А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         		А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
№
▓
%__inference_vgg16_layer_call_fn_17811
input_2!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@А
	unknown_4:	А%
	unknown_5:АА
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А&

unknown_11:АА

unknown_12:	А&

unknown_13:АА

unknown_14:	А&

unknown_15:АА

unknown_16:	А&

unknown_17:АА

unknown_18:	А&

unknown_19:АА

unknown_20:	А&

unknown_21:АА

unknown_22:	А&

unknown_23:АА

unknown_24:	А
identityИвStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_vgg16_layer_call_and_return_conditional_losses_17756x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:         ЦЦ: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:         ЦЦ
!
_user_specified_name	input_2
Т
Г
G__inference_block5_conv3_layer_call_and_return_conditional_losses_20025

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		АY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         		Аj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         		Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         		А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
▒
G
+__inference_block3_pool_layer_call_fn_19890

inputs
identity╫
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block3_pool_layer_call_and_return_conditional_losses_17495Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
О
b
F__inference_block2_pool_layer_call_and_return_conditional_losses_17483

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Т
Г
G__inference_block3_conv2_layer_call_and_return_conditional_losses_19865

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%АY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         %%Аj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         %%Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         %%А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         %%А
 
_user_specified_nameinputs
О
b
F__inference_block3_pool_layer_call_and_return_conditional_losses_17495

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
▒
G
+__inference_block4_pool_layer_call_fn_19960

inputs
identity╫
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block4_pool_layer_call_and_return_conditional_losses_17507Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
·
д
,__inference_block3_conv2_layer_call_fn_19854

inputs#
unknown:АА
	unknown_0:	А
identityИвStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         %%А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block3_conv2_layer_call_and_return_conditional_losses_17627x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         %%А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         %%А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         %%А
 
_user_specified_nameinputs
Т
Г
G__inference_block2_conv2_layer_call_and_return_conditional_losses_19815

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         KKА*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         KKАY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         KKАj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         KKАw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         KKА: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         KKА
 
_user_specified_nameinputs
▒
G
+__inference_block2_pool_layer_call_fn_19820

inputs
identity╫
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block2_pool_layer_call_and_return_conditional_losses_17483Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
О
В
G__inference_block2_conv1_layer_call_and_return_conditional_losses_17575

inputs9
conv2d_readvariableop_resource:@А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         KKА*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         KKАY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         KKАj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         KKАw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         KK@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         KK@
 
_user_specified_nameinputs
О
b
F__inference_block1_pool_layer_call_and_return_conditional_losses_17471

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Т
А
G__inference_block1_conv1_layer_call_and_return_conditional_losses_17540

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ы
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЦЦ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЦЦ@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:         ЦЦ@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:         ЦЦ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ЦЦ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ЦЦ
 
_user_specified_nameinputs
пЗ
ї
@__inference_vgg16_layer_call_and_return_conditional_losses_19674

inputsE
+block1_conv1_conv2d_readvariableop_resource:@:
,block1_conv1_biasadd_readvariableop_resource:@E
+block1_conv2_conv2d_readvariableop_resource:@@:
,block1_conv2_biasadd_readvariableop_resource:@F
+block2_conv1_conv2d_readvariableop_resource:@А;
,block2_conv1_biasadd_readvariableop_resource:	АG
+block2_conv2_conv2d_readvariableop_resource:АА;
,block2_conv2_biasadd_readvariableop_resource:	АG
+block3_conv1_conv2d_readvariableop_resource:АА;
,block3_conv1_biasadd_readvariableop_resource:	АG
+block3_conv2_conv2d_readvariableop_resource:АА;
,block3_conv2_biasadd_readvariableop_resource:	АG
+block3_conv3_conv2d_readvariableop_resource:АА;
,block3_conv3_biasadd_readvariableop_resource:	АG
+block4_conv1_conv2d_readvariableop_resource:АА;
,block4_conv1_biasadd_readvariableop_resource:	АG
+block4_conv2_conv2d_readvariableop_resource:АА;
,block4_conv2_biasadd_readvariableop_resource:	АG
+block4_conv3_conv2d_readvariableop_resource:АА;
,block4_conv3_biasadd_readvariableop_resource:	АG
+block5_conv1_conv2d_readvariableop_resource:АА;
,block5_conv1_biasadd_readvariableop_resource:	АG
+block5_conv2_conv2d_readvariableop_resource:АА;
,block5_conv2_biasadd_readvariableop_resource:	АG
+block5_conv3_conv2d_readvariableop_resource:АА;
,block5_conv3_biasadd_readvariableop_resource:	А
identityИв#block1_conv1/BiasAdd/ReadVariableOpв"block1_conv1/Conv2D/ReadVariableOpв#block1_conv2/BiasAdd/ReadVariableOpв"block1_conv2/Conv2D/ReadVariableOpв#block2_conv1/BiasAdd/ReadVariableOpв"block2_conv1/Conv2D/ReadVariableOpв#block2_conv2/BiasAdd/ReadVariableOpв"block2_conv2/Conv2D/ReadVariableOpв#block3_conv1/BiasAdd/ReadVariableOpв"block3_conv1/Conv2D/ReadVariableOpв#block3_conv2/BiasAdd/ReadVariableOpв"block3_conv2/Conv2D/ReadVariableOpв#block3_conv3/BiasAdd/ReadVariableOpв"block3_conv3/Conv2D/ReadVariableOpв#block4_conv1/BiasAdd/ReadVariableOpв"block4_conv1/Conv2D/ReadVariableOpв#block4_conv2/BiasAdd/ReadVariableOpв"block4_conv2/Conv2D/ReadVariableOpв#block4_conv3/BiasAdd/ReadVariableOpв"block4_conv3/Conv2D/ReadVariableOpв#block5_conv1/BiasAdd/ReadVariableOpв"block5_conv1/Conv2D/ReadVariableOpв#block5_conv2/BiasAdd/ReadVariableOpв"block5_conv2/Conv2D/ReadVariableOpв#block5_conv3/BiasAdd/ReadVariableOpв"block5_conv3/Conv2D/ReadVariableOpЦ
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp+block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0╡
block1_conv1/Conv2DConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЦЦ@*
paddingSAME*
strides
М
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp,block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ж
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЦЦ@t
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:         ЦЦ@Ц
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp+block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0╬
block1_conv2/Conv2DConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЦЦ@*
paddingSAME*
strides
М
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp,block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ж
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЦЦ@t
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:         ЦЦ@м
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*/
_output_shapes
:         KK@*
ksize
*
paddingVALID*
strides
Ч
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0╩
block2_conv1/Conv2DConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         KKА*
paddingSAME*
strides
Н
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0е
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         KKАs
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:         KKАШ
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0═
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         KKА*
paddingSAME*
strides
Н
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0е
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         KKАs
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:         KKАн
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:         %%А*
ksize
*
paddingVALID*
strides
Ш
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp+block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0╩
block3_conv1/Conv2DConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
Н
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp,block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0е
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%Аs
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:         %%АШ
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp+block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0═
block3_conv2/Conv2DConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
Н
#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp,block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0е
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%Аs
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:         %%АШ
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp+block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0═
block3_conv3/Conv2DConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
Н
#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp,block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0е
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%Аs
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:         %%Ан
block3_pool/MaxPoolMaxPoolblock3_conv3/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
Ш
"block4_conv1/Conv2D/ReadVariableOpReadVariableOp+block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0╩
block4_conv1/Conv2DConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
Н
#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp,block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0е
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Аs
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:         АШ
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp+block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0═
block4_conv2/Conv2DConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
Н
#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp,block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0е
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Аs
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:         АШ
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp+block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0═
block4_conv3/Conv2DConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
Н
#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp,block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0е
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         Аs
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:         Ан
block4_pool/MaxPoolMaxPoolblock4_conv3/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
Ш
"block5_conv1/Conv2D/ReadVariableOpReadVariableOp+block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0╩
block5_conv1/Conv2DConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
Н
#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp,block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0е
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		Аs
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:         		АШ
"block5_conv2/Conv2D/ReadVariableOpReadVariableOp+block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0═
block5_conv2/Conv2DConv2Dblock5_conv1/Relu:activations:0*block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
Н
#block5_conv2/BiasAdd/ReadVariableOpReadVariableOp,block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0е
block5_conv2/BiasAddBiasAddblock5_conv2/Conv2D:output:0+block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		Аs
block5_conv2/ReluRelublock5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:         		АШ
"block5_conv3/Conv2D/ReadVariableOpReadVariableOp+block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0═
block5_conv3/Conv2DConv2Dblock5_conv2/Relu:activations:0*block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
Н
#block5_conv3/BiasAdd/ReadVariableOpReadVariableOp,block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0е
block5_conv3/BiasAddBiasAddblock5_conv3/Conv2D:output:0+block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		Аs
block5_conv3/ReluRelublock5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:         		Ан
block5_pool/MaxPoolMaxPoolblock5_conv3/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
t
IdentityIdentityblock5_pool/MaxPool:output:0^NoOp*
T0*0
_output_shapes
:         АХ
NoOpNoOp$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp$^block5_conv2/BiasAdd/ReadVariableOp#^block5_conv2/Conv2D/ReadVariableOp$^block5_conv3/BiasAdd/ReadVariableOp#^block5_conv3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:         ЦЦ: : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp2J
#block4_conv2/BiasAdd/ReadVariableOp#block4_conv2/BiasAdd/ReadVariableOp2H
"block4_conv2/Conv2D/ReadVariableOp"block4_conv2/Conv2D/ReadVariableOp2J
#block4_conv3/BiasAdd/ReadVariableOp#block4_conv3/BiasAdd/ReadVariableOp2H
"block4_conv3/Conv2D/ReadVariableOp"block4_conv3/Conv2D/ReadVariableOp2J
#block5_conv1/BiasAdd/ReadVariableOp#block5_conv1/BiasAdd/ReadVariableOp2H
"block5_conv1/Conv2D/ReadVariableOp"block5_conv1/Conv2D/ReadVariableOp2J
#block5_conv2/BiasAdd/ReadVariableOp#block5_conv2/BiasAdd/ReadVariableOp2H
"block5_conv2/Conv2D/ReadVariableOp"block5_conv2/Conv2D/ReadVariableOp2J
#block5_conv3/BiasAdd/ReadVariableOp#block5_conv3/BiasAdd/ReadVariableOp2H
"block5_conv3/Conv2D/ReadVariableOp"block5_conv3/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:         ЦЦ
 
_user_specified_nameinputs
й
м
,__inference_sequential_1_layer_call_fn_19063

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@А
	unknown_4:	А%
	unknown_5:АА
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А&

unknown_11:АА

unknown_12:	А&

unknown_13:АА

unknown_14:	А&

unknown_15:АА

unknown_16:	А&

unknown_17:АА

unknown_18:	А&

unknown_19:АА

unknown_20:	А&

unknown_21:АА

unknown_22:	А&

unknown_23:АА

unknown_24:	А

unknown_25:
А@А

unknown_26:	А

unknown_27:	А

unknown_28:
identityИвStatefulPartitionedCall▐
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_18437o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:         ЦЦ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ЦЦ
 
_user_specified_nameinputs
Т
Г
G__inference_block3_conv1_layer_call_and_return_conditional_losses_17610

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%АY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         %%Аj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         %%Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         %%А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         %%А
 
_user_specified_nameinputs
Ь

Ї
B__inference_dense_3_layer_call_and_return_conditional_losses_19725

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
гз
Ь
G__inference_sequential_1_layer_call_and_return_conditional_losses_19360

inputsK
1vgg16_block1_conv1_conv2d_readvariableop_resource:@@
2vgg16_block1_conv1_biasadd_readvariableop_resource:@K
1vgg16_block1_conv2_conv2d_readvariableop_resource:@@@
2vgg16_block1_conv2_biasadd_readvariableop_resource:@L
1vgg16_block2_conv1_conv2d_readvariableop_resource:@АA
2vgg16_block2_conv1_biasadd_readvariableop_resource:	АM
1vgg16_block2_conv2_conv2d_readvariableop_resource:ААA
2vgg16_block2_conv2_biasadd_readvariableop_resource:	АM
1vgg16_block3_conv1_conv2d_readvariableop_resource:ААA
2vgg16_block3_conv1_biasadd_readvariableop_resource:	АM
1vgg16_block3_conv2_conv2d_readvariableop_resource:ААA
2vgg16_block3_conv2_biasadd_readvariableop_resource:	АM
1vgg16_block3_conv3_conv2d_readvariableop_resource:ААA
2vgg16_block3_conv3_biasadd_readvariableop_resource:	АM
1vgg16_block4_conv1_conv2d_readvariableop_resource:ААA
2vgg16_block4_conv1_biasadd_readvariableop_resource:	АM
1vgg16_block4_conv2_conv2d_readvariableop_resource:ААA
2vgg16_block4_conv2_biasadd_readvariableop_resource:	АM
1vgg16_block4_conv3_conv2d_readvariableop_resource:ААA
2vgg16_block4_conv3_biasadd_readvariableop_resource:	АM
1vgg16_block5_conv1_conv2d_readvariableop_resource:ААA
2vgg16_block5_conv1_biasadd_readvariableop_resource:	АM
1vgg16_block5_conv2_conv2d_readvariableop_resource:ААA
2vgg16_block5_conv2_biasadd_readvariableop_resource:	АM
1vgg16_block5_conv3_conv2d_readvariableop_resource:ААA
2vgg16_block5_conv3_biasadd_readvariableop_resource:	А:
&dense_2_matmul_readvariableop_resource:
А@А6
'dense_2_biasadd_readvariableop_resource:	А9
&dense_3_matmul_readvariableop_resource:	А5
'dense_3_biasadd_readvariableop_resource:
identityИвdense_2/BiasAdd/ReadVariableOpвdense_2/MatMul/ReadVariableOpвdense_3/BiasAdd/ReadVariableOpвdense_3/MatMul/ReadVariableOpв)vgg16/block1_conv1/BiasAdd/ReadVariableOpв(vgg16/block1_conv1/Conv2D/ReadVariableOpв)vgg16/block1_conv2/BiasAdd/ReadVariableOpв(vgg16/block1_conv2/Conv2D/ReadVariableOpв)vgg16/block2_conv1/BiasAdd/ReadVariableOpв(vgg16/block2_conv1/Conv2D/ReadVariableOpв)vgg16/block2_conv2/BiasAdd/ReadVariableOpв(vgg16/block2_conv2/Conv2D/ReadVariableOpв)vgg16/block3_conv1/BiasAdd/ReadVariableOpв(vgg16/block3_conv1/Conv2D/ReadVariableOpв)vgg16/block3_conv2/BiasAdd/ReadVariableOpв(vgg16/block3_conv2/Conv2D/ReadVariableOpв)vgg16/block3_conv3/BiasAdd/ReadVariableOpв(vgg16/block3_conv3/Conv2D/ReadVariableOpв)vgg16/block4_conv1/BiasAdd/ReadVariableOpв(vgg16/block4_conv1/Conv2D/ReadVariableOpв)vgg16/block4_conv2/BiasAdd/ReadVariableOpв(vgg16/block4_conv2/Conv2D/ReadVariableOpв)vgg16/block4_conv3/BiasAdd/ReadVariableOpв(vgg16/block4_conv3/Conv2D/ReadVariableOpв)vgg16/block5_conv1/BiasAdd/ReadVariableOpв(vgg16/block5_conv1/Conv2D/ReadVariableOpв)vgg16/block5_conv2/BiasAdd/ReadVariableOpв(vgg16/block5_conv2/Conv2D/ReadVariableOpв)vgg16/block5_conv3/BiasAdd/ReadVariableOpв(vgg16/block5_conv3/Conv2D/ReadVariableOpв
(vgg16/block1_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0┴
vgg16/block1_conv1/Conv2DConv2Dinputs0vgg16/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЦЦ@*
paddingSAME*
strides
Ш
)vgg16/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╕
vgg16/block1_conv1/BiasAddBiasAdd"vgg16/block1_conv1/Conv2D:output:01vgg16/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЦЦ@А
vgg16/block1_conv1/ReluRelu#vgg16/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:         ЦЦ@в
(vgg16/block1_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0р
vgg16/block1_conv2/Conv2DConv2D%vgg16/block1_conv1/Relu:activations:00vgg16/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЦЦ@*
paddingSAME*
strides
Ш
)vgg16/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╕
vgg16/block1_conv2/BiasAddBiasAdd"vgg16/block1_conv2/Conv2D:output:01vgg16/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЦЦ@А
vgg16/block1_conv2/ReluRelu#vgg16/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:         ЦЦ@╕
vgg16/block1_pool/MaxPoolMaxPool%vgg16/block1_conv2/Relu:activations:0*/
_output_shapes
:         KK@*
ksize
*
paddingVALID*
strides
г
(vgg16/block2_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0▄
vgg16/block2_conv1/Conv2DConv2D"vgg16/block1_pool/MaxPool:output:00vgg16/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         KKА*
paddingSAME*
strides
Щ
)vgg16/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
vgg16/block2_conv1/BiasAddBiasAdd"vgg16/block2_conv1/Conv2D:output:01vgg16/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         KKА
vgg16/block2_conv1/ReluRelu#vgg16/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:         KKАд
(vgg16/block2_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0▀
vgg16/block2_conv2/Conv2DConv2D%vgg16/block2_conv1/Relu:activations:00vgg16/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         KKА*
paddingSAME*
strides
Щ
)vgg16/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
vgg16/block2_conv2/BiasAddBiasAdd"vgg16/block2_conv2/Conv2D:output:01vgg16/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         KKА
vgg16/block2_conv2/ReluRelu#vgg16/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:         KKА╣
vgg16/block2_pool/MaxPoolMaxPool%vgg16/block2_conv2/Relu:activations:0*0
_output_shapes
:         %%А*
ksize
*
paddingVALID*
strides
д
(vgg16/block3_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0▄
vgg16/block3_conv1/Conv2DConv2D"vgg16/block2_pool/MaxPool:output:00vgg16/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
Щ
)vgg16/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
vgg16/block3_conv1/BiasAddBiasAdd"vgg16/block3_conv1/Conv2D:output:01vgg16/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А
vgg16/block3_conv1/ReluRelu#vgg16/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:         %%Ад
(vgg16/block3_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0▀
vgg16/block3_conv2/Conv2DConv2D%vgg16/block3_conv1/Relu:activations:00vgg16/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
Щ
)vgg16/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
vgg16/block3_conv2/BiasAddBiasAdd"vgg16/block3_conv2/Conv2D:output:01vgg16/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А
vgg16/block3_conv2/ReluRelu#vgg16/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:         %%Ад
(vgg16/block3_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg16_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0▀
vgg16/block3_conv3/Conv2DConv2D%vgg16/block3_conv2/Relu:activations:00vgg16/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
Щ
)vgg16/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
vgg16/block3_conv3/BiasAddBiasAdd"vgg16/block3_conv3/Conv2D:output:01vgg16/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А
vgg16/block3_conv3/ReluRelu#vgg16/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А╣
vgg16/block3_pool/MaxPoolMaxPool%vgg16/block3_conv3/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
д
(vgg16/block4_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0▄
vgg16/block4_conv1/Conv2DConv2D"vgg16/block3_pool/MaxPool:output:00vgg16/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
Щ
)vgg16/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
vgg16/block4_conv1/BiasAddBiasAdd"vgg16/block4_conv1/Conv2D:output:01vgg16/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А
vgg16/block4_conv1/ReluRelu#vgg16/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:         Ад
(vgg16/block4_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0▀
vgg16/block4_conv2/Conv2DConv2D%vgg16/block4_conv1/Relu:activations:00vgg16/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
Щ
)vgg16/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
vgg16/block4_conv2/BiasAddBiasAdd"vgg16/block4_conv2/Conv2D:output:01vgg16/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А
vgg16/block4_conv2/ReluRelu#vgg16/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:         Ад
(vgg16/block4_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg16_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0▀
vgg16/block4_conv3/Conv2DConv2D%vgg16/block4_conv2/Relu:activations:00vgg16/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
Щ
)vgg16/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
vgg16/block4_conv3/BiasAddBiasAdd"vgg16/block4_conv3/Conv2D:output:01vgg16/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А
vgg16/block4_conv3/ReluRelu#vgg16/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:         А╣
vgg16/block4_pool/MaxPoolMaxPool%vgg16/block4_conv3/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
д
(vgg16/block5_conv1/Conv2D/ReadVariableOpReadVariableOp1vgg16_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0▄
vgg16/block5_conv1/Conv2DConv2D"vgg16/block4_pool/MaxPool:output:00vgg16/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
Щ
)vgg16/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
vgg16/block5_conv1/BiasAddBiasAdd"vgg16/block5_conv1/Conv2D:output:01vgg16/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А
vgg16/block5_conv1/ReluRelu#vgg16/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:         		Ад
(vgg16/block5_conv2/Conv2D/ReadVariableOpReadVariableOp1vgg16_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0▀
vgg16/block5_conv2/Conv2DConv2D%vgg16/block5_conv1/Relu:activations:00vgg16/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
Щ
)vgg16/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
vgg16/block5_conv2/BiasAddBiasAdd"vgg16/block5_conv2/Conv2D:output:01vgg16/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А
vgg16/block5_conv2/ReluRelu#vgg16/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:         		Ад
(vgg16/block5_conv3/Conv2D/ReadVariableOpReadVariableOp1vgg16_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0▀
vgg16/block5_conv3/Conv2DConv2D%vgg16/block5_conv2/Relu:activations:00vgg16/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
Щ
)vgg16/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp2vgg16_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╖
vgg16/block5_conv3/BiasAddBiasAdd"vgg16/block5_conv3/Conv2D:output:01vgg16/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А
vgg16/block5_conv3/ReluRelu#vgg16/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:         		А╣
vgg16/block5_pool/MaxPoolMaxPool%vgg16/block5_conv3/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"        Н
flatten_1/ReshapeReshape"vgg16/block5_pool/MaxPool:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:         А@Ж
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype0О
dense_2/MatMulMatMulflatten_1/Reshape:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АГ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0П
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аa
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:         АЕ
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Н
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         В
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         f
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:         b
IdentityIdentitydense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:         │

NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*^vgg16/block1_conv1/BiasAdd/ReadVariableOp)^vgg16/block1_conv1/Conv2D/ReadVariableOp*^vgg16/block1_conv2/BiasAdd/ReadVariableOp)^vgg16/block1_conv2/Conv2D/ReadVariableOp*^vgg16/block2_conv1/BiasAdd/ReadVariableOp)^vgg16/block2_conv1/Conv2D/ReadVariableOp*^vgg16/block2_conv2/BiasAdd/ReadVariableOp)^vgg16/block2_conv2/Conv2D/ReadVariableOp*^vgg16/block3_conv1/BiasAdd/ReadVariableOp)^vgg16/block3_conv1/Conv2D/ReadVariableOp*^vgg16/block3_conv2/BiasAdd/ReadVariableOp)^vgg16/block3_conv2/Conv2D/ReadVariableOp*^vgg16/block3_conv3/BiasAdd/ReadVariableOp)^vgg16/block3_conv3/Conv2D/ReadVariableOp*^vgg16/block4_conv1/BiasAdd/ReadVariableOp)^vgg16/block4_conv1/Conv2D/ReadVariableOp*^vgg16/block4_conv2/BiasAdd/ReadVariableOp)^vgg16/block4_conv2/Conv2D/ReadVariableOp*^vgg16/block4_conv3/BiasAdd/ReadVariableOp)^vgg16/block4_conv3/Conv2D/ReadVariableOp*^vgg16/block5_conv1/BiasAdd/ReadVariableOp)^vgg16/block5_conv1/Conv2D/ReadVariableOp*^vgg16/block5_conv2/BiasAdd/ReadVariableOp)^vgg16/block5_conv2/Conv2D/ReadVariableOp*^vgg16/block5_conv3/BiasAdd/ReadVariableOp)^vgg16/block5_conv3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:         ЦЦ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2V
)vgg16/block1_conv1/BiasAdd/ReadVariableOp)vgg16/block1_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block1_conv1/Conv2D/ReadVariableOp(vgg16/block1_conv1/Conv2D/ReadVariableOp2V
)vgg16/block1_conv2/BiasAdd/ReadVariableOp)vgg16/block1_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block1_conv2/Conv2D/ReadVariableOp(vgg16/block1_conv2/Conv2D/ReadVariableOp2V
)vgg16/block2_conv1/BiasAdd/ReadVariableOp)vgg16/block2_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block2_conv1/Conv2D/ReadVariableOp(vgg16/block2_conv1/Conv2D/ReadVariableOp2V
)vgg16/block2_conv2/BiasAdd/ReadVariableOp)vgg16/block2_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block2_conv2/Conv2D/ReadVariableOp(vgg16/block2_conv2/Conv2D/ReadVariableOp2V
)vgg16/block3_conv1/BiasAdd/ReadVariableOp)vgg16/block3_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block3_conv1/Conv2D/ReadVariableOp(vgg16/block3_conv1/Conv2D/ReadVariableOp2V
)vgg16/block3_conv2/BiasAdd/ReadVariableOp)vgg16/block3_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block3_conv2/Conv2D/ReadVariableOp(vgg16/block3_conv2/Conv2D/ReadVariableOp2V
)vgg16/block3_conv3/BiasAdd/ReadVariableOp)vgg16/block3_conv3/BiasAdd/ReadVariableOp2T
(vgg16/block3_conv3/Conv2D/ReadVariableOp(vgg16/block3_conv3/Conv2D/ReadVariableOp2V
)vgg16/block4_conv1/BiasAdd/ReadVariableOp)vgg16/block4_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block4_conv1/Conv2D/ReadVariableOp(vgg16/block4_conv1/Conv2D/ReadVariableOp2V
)vgg16/block4_conv2/BiasAdd/ReadVariableOp)vgg16/block4_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block4_conv2/Conv2D/ReadVariableOp(vgg16/block4_conv2/Conv2D/ReadVariableOp2V
)vgg16/block4_conv3/BiasAdd/ReadVariableOp)vgg16/block4_conv3/BiasAdd/ReadVariableOp2T
(vgg16/block4_conv3/Conv2D/ReadVariableOp(vgg16/block4_conv3/Conv2D/ReadVariableOp2V
)vgg16/block5_conv1/BiasAdd/ReadVariableOp)vgg16/block5_conv1/BiasAdd/ReadVariableOp2T
(vgg16/block5_conv1/Conv2D/ReadVariableOp(vgg16/block5_conv1/Conv2D/ReadVariableOp2V
)vgg16/block5_conv2/BiasAdd/ReadVariableOp)vgg16/block5_conv2/BiasAdd/ReadVariableOp2T
(vgg16/block5_conv2/Conv2D/ReadVariableOp(vgg16/block5_conv2/Conv2D/ReadVariableOp2V
)vgg16/block5_conv3/BiasAdd/ReadVariableOp)vgg16/block5_conv3/BiasAdd/ReadVariableOp2T
(vgg16/block5_conv3/Conv2D/ReadVariableOp(vgg16/block5_conv3/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:         ЦЦ
 
_user_specified_nameinputs
√
б
,__inference_block1_conv1_layer_call_fn_19734

inputs!
unknown:@
	unknown_0:@
identityИвStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЦЦ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block1_conv1_layer_call_and_return_conditional_losses_17540y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЦЦ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ЦЦ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ЦЦ
 
_user_specified_nameinputs
Т
А
G__inference_block1_conv1_layer_call_and_return_conditional_losses_19745

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ы
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЦЦ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЦЦ@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:         ЦЦ@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:         ЦЦ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ЦЦ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ЦЦ
 
_user_specified_nameinputs
№
▓
%__inference_vgg16_layer_call_fn_18186
input_2!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@А
	unknown_4:	А%
	unknown_5:АА
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А&

unknown_11:АА

unknown_12:	А&

unknown_13:АА

unknown_14:	А&

unknown_15:АА

unknown_16:	А&

unknown_17:АА

unknown_18:	А&

unknown_19:АА

unknown_20:	А&

unknown_21:АА

unknown_22:	А&

unknown_23:АА

unknown_24:	А
identityИвStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_vgg16_layer_call_and_return_conditional_losses_18074x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:         ЦЦ: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:         ЦЦ
!
_user_specified_name	input_2
л
╔	
G__inference_sequential_1_layer_call_and_return_conditional_losses_18857
vgg16_input%
vgg16_18792:@
vgg16_18794:@%
vgg16_18796:@@
vgg16_18798:@&
vgg16_18800:@А
vgg16_18802:	А'
vgg16_18804:АА
vgg16_18806:	А'
vgg16_18808:АА
vgg16_18810:	А'
vgg16_18812:АА
vgg16_18814:	А'
vgg16_18816:АА
vgg16_18818:	А'
vgg16_18820:АА
vgg16_18822:	А'
vgg16_18824:АА
vgg16_18826:	А'
vgg16_18828:АА
vgg16_18830:	А'
vgg16_18832:АА
vgg16_18834:	А'
vgg16_18836:АА
vgg16_18838:	А'
vgg16_18840:АА
vgg16_18842:	А!
dense_2_18846:
А@А
dense_2_18848:	А 
dense_3_18851:	А
dense_3_18853:
identityИвdense_2/StatefulPartitionedCallвdense_3/StatefulPartitionedCallвvgg16/StatefulPartitionedCall┌
vgg16/StatefulPartitionedCallStatefulPartitionedCallvgg16_inputvgg16_18792vgg16_18794vgg16_18796vgg16_18798vgg16_18800vgg16_18802vgg16_18804vgg16_18806vgg16_18808vgg16_18810vgg16_18812vgg16_18814vgg16_18816vgg16_18818vgg16_18820vgg16_18822vgg16_18824vgg16_18826vgg16_18828vgg16_18830vgg16_18832vgg16_18834vgg16_18836vgg16_18838vgg16_18840vgg16_18842*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_vgg16_layer_call_and_return_conditional_losses_17756▌
flatten_1/PartitionedCallPartitionedCall&vgg16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_18400Й
dense_2/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_2_18846dense_2_18848*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_18413О
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_18851dense_3_18853*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_18430w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         к
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall^vgg16/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:         ЦЦ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2>
vgg16/StatefulPartitionedCallvgg16/StatefulPartitionedCall:^ Z
1
_output_shapes
:         ЦЦ
%
_user_specified_namevgg16_input
Т
Г
G__inference_block4_conv3_layer_call_and_return_conditional_losses_17696

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         Аj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
Т
Г
G__inference_block3_conv1_layer_call_and_return_conditional_losses_19845

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%АY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         %%Аj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         %%Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         %%А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         %%А
 
_user_specified_nameinputs
И
и
#__inference_signature_wrapper_18998
vgg16_input!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@А
	unknown_4:	А%
	unknown_5:АА
	unknown_6:	А%
	unknown_7:АА
	unknown_8:	А%
	unknown_9:АА

unknown_10:	А&

unknown_11:АА

unknown_12:	А&

unknown_13:АА

unknown_14:	А&

unknown_15:АА

unknown_16:	А&

unknown_17:АА

unknown_18:	А&

unknown_19:АА

unknown_20:	А&

unknown_21:АА

unknown_22:	А&

unknown_23:АА

unknown_24:	А

unknown_25:
А@А

unknown_26:	А

unknown_27:	А

unknown_28:
identityИвStatefulPartitionedCall╝
StatefulPartitionedCallStatefulPartitionedCallvgg16_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__wrapped_model_17462o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:         ЦЦ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:         ЦЦ
%
_user_specified_namevgg16_input
П╩
Ж!
 __inference__wrapped_model_17462
vgg16_inputX
>sequential_1_vgg16_block1_conv1_conv2d_readvariableop_resource:@M
?sequential_1_vgg16_block1_conv1_biasadd_readvariableop_resource:@X
>sequential_1_vgg16_block1_conv2_conv2d_readvariableop_resource:@@M
?sequential_1_vgg16_block1_conv2_biasadd_readvariableop_resource:@Y
>sequential_1_vgg16_block2_conv1_conv2d_readvariableop_resource:@АN
?sequential_1_vgg16_block2_conv1_biasadd_readvariableop_resource:	АZ
>sequential_1_vgg16_block2_conv2_conv2d_readvariableop_resource:ААN
?sequential_1_vgg16_block2_conv2_biasadd_readvariableop_resource:	АZ
>sequential_1_vgg16_block3_conv1_conv2d_readvariableop_resource:ААN
?sequential_1_vgg16_block3_conv1_biasadd_readvariableop_resource:	АZ
>sequential_1_vgg16_block3_conv2_conv2d_readvariableop_resource:ААN
?sequential_1_vgg16_block3_conv2_biasadd_readvariableop_resource:	АZ
>sequential_1_vgg16_block3_conv3_conv2d_readvariableop_resource:ААN
?sequential_1_vgg16_block3_conv3_biasadd_readvariableop_resource:	АZ
>sequential_1_vgg16_block4_conv1_conv2d_readvariableop_resource:ААN
?sequential_1_vgg16_block4_conv1_biasadd_readvariableop_resource:	АZ
>sequential_1_vgg16_block4_conv2_conv2d_readvariableop_resource:ААN
?sequential_1_vgg16_block4_conv2_biasadd_readvariableop_resource:	АZ
>sequential_1_vgg16_block4_conv3_conv2d_readvariableop_resource:ААN
?sequential_1_vgg16_block4_conv3_biasadd_readvariableop_resource:	АZ
>sequential_1_vgg16_block5_conv1_conv2d_readvariableop_resource:ААN
?sequential_1_vgg16_block5_conv1_biasadd_readvariableop_resource:	АZ
>sequential_1_vgg16_block5_conv2_conv2d_readvariableop_resource:ААN
?sequential_1_vgg16_block5_conv2_biasadd_readvariableop_resource:	АZ
>sequential_1_vgg16_block5_conv3_conv2d_readvariableop_resource:ААN
?sequential_1_vgg16_block5_conv3_biasadd_readvariableop_resource:	АG
3sequential_1_dense_2_matmul_readvariableop_resource:
А@АC
4sequential_1_dense_2_biasadd_readvariableop_resource:	АF
3sequential_1_dense_3_matmul_readvariableop_resource:	АB
4sequential_1_dense_3_biasadd_readvariableop_resource:
identityИв+sequential_1/dense_2/BiasAdd/ReadVariableOpв*sequential_1/dense_2/MatMul/ReadVariableOpв+sequential_1/dense_3/BiasAdd/ReadVariableOpв*sequential_1/dense_3/MatMul/ReadVariableOpв6sequential_1/vgg16/block1_conv1/BiasAdd/ReadVariableOpв5sequential_1/vgg16/block1_conv1/Conv2D/ReadVariableOpв6sequential_1/vgg16/block1_conv2/BiasAdd/ReadVariableOpв5sequential_1/vgg16/block1_conv2/Conv2D/ReadVariableOpв6sequential_1/vgg16/block2_conv1/BiasAdd/ReadVariableOpв5sequential_1/vgg16/block2_conv1/Conv2D/ReadVariableOpв6sequential_1/vgg16/block2_conv2/BiasAdd/ReadVariableOpв5sequential_1/vgg16/block2_conv2/Conv2D/ReadVariableOpв6sequential_1/vgg16/block3_conv1/BiasAdd/ReadVariableOpв5sequential_1/vgg16/block3_conv1/Conv2D/ReadVariableOpв6sequential_1/vgg16/block3_conv2/BiasAdd/ReadVariableOpв5sequential_1/vgg16/block3_conv2/Conv2D/ReadVariableOpв6sequential_1/vgg16/block3_conv3/BiasAdd/ReadVariableOpв5sequential_1/vgg16/block3_conv3/Conv2D/ReadVariableOpв6sequential_1/vgg16/block4_conv1/BiasAdd/ReadVariableOpв5sequential_1/vgg16/block4_conv1/Conv2D/ReadVariableOpв6sequential_1/vgg16/block4_conv2/BiasAdd/ReadVariableOpв5sequential_1/vgg16/block4_conv2/Conv2D/ReadVariableOpв6sequential_1/vgg16/block4_conv3/BiasAdd/ReadVariableOpв5sequential_1/vgg16/block4_conv3/Conv2D/ReadVariableOpв6sequential_1/vgg16/block5_conv1/BiasAdd/ReadVariableOpв5sequential_1/vgg16/block5_conv1/Conv2D/ReadVariableOpв6sequential_1/vgg16/block5_conv2/BiasAdd/ReadVariableOpв5sequential_1/vgg16/block5_conv2/Conv2D/ReadVariableOpв6sequential_1/vgg16/block5_conv3/BiasAdd/ReadVariableOpв5sequential_1/vgg16/block5_conv3/Conv2D/ReadVariableOp╝
5sequential_1/vgg16/block1_conv1/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg16_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0р
&sequential_1/vgg16/block1_conv1/Conv2DConv2Dvgg16_input=sequential_1/vgg16/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЦЦ@*
paddingSAME*
strides
▓
6sequential_1/vgg16/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg16_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0▀
'sequential_1/vgg16/block1_conv1/BiasAddBiasAdd/sequential_1/vgg16/block1_conv1/Conv2D:output:0>sequential_1/vgg16/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЦЦ@Ъ
$sequential_1/vgg16/block1_conv1/ReluRelu0sequential_1/vgg16/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:         ЦЦ@╝
5sequential_1/vgg16/block1_conv2/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg16_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0З
&sequential_1/vgg16/block1_conv2/Conv2DConv2D2sequential_1/vgg16/block1_conv1/Relu:activations:0=sequential_1/vgg16/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЦЦ@*
paddingSAME*
strides
▓
6sequential_1/vgg16/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg16_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0▀
'sequential_1/vgg16/block1_conv2/BiasAddBiasAdd/sequential_1/vgg16/block1_conv2/Conv2D:output:0>sequential_1/vgg16/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЦЦ@Ъ
$sequential_1/vgg16/block1_conv2/ReluRelu0sequential_1/vgg16/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:         ЦЦ@╥
&sequential_1/vgg16/block1_pool/MaxPoolMaxPool2sequential_1/vgg16/block1_conv2/Relu:activations:0*/
_output_shapes
:         KK@*
ksize
*
paddingVALID*
strides
╜
5sequential_1/vgg16/block2_conv1/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg16_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0Г
&sequential_1/vgg16/block2_conv1/Conv2DConv2D/sequential_1/vgg16/block1_pool/MaxPool:output:0=sequential_1/vgg16/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         KKА*
paddingSAME*
strides
│
6sequential_1/vgg16/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg16_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0▐
'sequential_1/vgg16/block2_conv1/BiasAddBiasAdd/sequential_1/vgg16/block2_conv1/Conv2D:output:0>sequential_1/vgg16/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         KKАЩ
$sequential_1/vgg16/block2_conv1/ReluRelu0sequential_1/vgg16/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:         KKА╛
5sequential_1/vgg16/block2_conv2/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg16_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ж
&sequential_1/vgg16/block2_conv2/Conv2DConv2D2sequential_1/vgg16/block2_conv1/Relu:activations:0=sequential_1/vgg16/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         KKА*
paddingSAME*
strides
│
6sequential_1/vgg16/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg16_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0▐
'sequential_1/vgg16/block2_conv2/BiasAddBiasAdd/sequential_1/vgg16/block2_conv2/Conv2D:output:0>sequential_1/vgg16/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         KKАЩ
$sequential_1/vgg16/block2_conv2/ReluRelu0sequential_1/vgg16/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:         KKА╙
&sequential_1/vgg16/block2_pool/MaxPoolMaxPool2sequential_1/vgg16/block2_conv2/Relu:activations:0*0
_output_shapes
:         %%А*
ksize
*
paddingVALID*
strides
╛
5sequential_1/vgg16/block3_conv1/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg16_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Г
&sequential_1/vgg16/block3_conv1/Conv2DConv2D/sequential_1/vgg16/block2_pool/MaxPool:output:0=sequential_1/vgg16/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
│
6sequential_1/vgg16/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg16_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0▐
'sequential_1/vgg16/block3_conv1/BiasAddBiasAdd/sequential_1/vgg16/block3_conv1/Conv2D:output:0>sequential_1/vgg16/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%АЩ
$sequential_1/vgg16/block3_conv1/ReluRelu0sequential_1/vgg16/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А╛
5sequential_1/vgg16/block3_conv2/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg16_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ж
&sequential_1/vgg16/block3_conv2/Conv2DConv2D2sequential_1/vgg16/block3_conv1/Relu:activations:0=sequential_1/vgg16/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
│
6sequential_1/vgg16/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg16_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0▐
'sequential_1/vgg16/block3_conv2/BiasAddBiasAdd/sequential_1/vgg16/block3_conv2/Conv2D:output:0>sequential_1/vgg16/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%АЩ
$sequential_1/vgg16/block3_conv2/ReluRelu0sequential_1/vgg16/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А╛
5sequential_1/vgg16/block3_conv3/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg16_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ж
&sequential_1/vgg16/block3_conv3/Conv2DConv2D2sequential_1/vgg16/block3_conv2/Relu:activations:0=sequential_1/vgg16/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%А*
paddingSAME*
strides
│
6sequential_1/vgg16/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg16_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0▐
'sequential_1/vgg16/block3_conv3/BiasAddBiasAdd/sequential_1/vgg16/block3_conv3/Conv2D:output:0>sequential_1/vgg16/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         %%АЩ
$sequential_1/vgg16/block3_conv3/ReluRelu0sequential_1/vgg16/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:         %%А╙
&sequential_1/vgg16/block3_pool/MaxPoolMaxPool2sequential_1/vgg16/block3_conv3/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
╛
5sequential_1/vgg16/block4_conv1/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg16_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Г
&sequential_1/vgg16/block4_conv1/Conv2DConv2D/sequential_1/vgg16/block3_pool/MaxPool:output:0=sequential_1/vgg16/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
│
6sequential_1/vgg16/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg16_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0▐
'sequential_1/vgg16/block4_conv1/BiasAddBiasAdd/sequential_1/vgg16/block4_conv1/Conv2D:output:0>sequential_1/vgg16/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АЩ
$sequential_1/vgg16/block4_conv1/ReluRelu0sequential_1/vgg16/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:         А╛
5sequential_1/vgg16/block4_conv2/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg16_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ж
&sequential_1/vgg16/block4_conv2/Conv2DConv2D2sequential_1/vgg16/block4_conv1/Relu:activations:0=sequential_1/vgg16/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
│
6sequential_1/vgg16/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg16_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0▐
'sequential_1/vgg16/block4_conv2/BiasAddBiasAdd/sequential_1/vgg16/block4_conv2/Conv2D:output:0>sequential_1/vgg16/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АЩ
$sequential_1/vgg16/block4_conv2/ReluRelu0sequential_1/vgg16/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:         А╛
5sequential_1/vgg16/block4_conv3/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg16_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ж
&sequential_1/vgg16/block4_conv3/Conv2DConv2D2sequential_1/vgg16/block4_conv2/Relu:activations:0=sequential_1/vgg16/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingSAME*
strides
│
6sequential_1/vgg16/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg16_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0▐
'sequential_1/vgg16/block4_conv3/BiasAddBiasAdd/sequential_1/vgg16/block4_conv3/Conv2D:output:0>sequential_1/vgg16/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         АЩ
$sequential_1/vgg16/block4_conv3/ReluRelu0sequential_1/vgg16/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:         А╙
&sequential_1/vgg16/block4_pool/MaxPoolMaxPool2sequential_1/vgg16/block4_conv3/Relu:activations:0*0
_output_shapes
:         		А*
ksize
*
paddingVALID*
strides
╛
5sequential_1/vgg16/block5_conv1/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg16_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Г
&sequential_1/vgg16/block5_conv1/Conv2DConv2D/sequential_1/vgg16/block4_pool/MaxPool:output:0=sequential_1/vgg16/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
│
6sequential_1/vgg16/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg16_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0▐
'sequential_1/vgg16/block5_conv1/BiasAddBiasAdd/sequential_1/vgg16/block5_conv1/Conv2D:output:0>sequential_1/vgg16/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		АЩ
$sequential_1/vgg16/block5_conv1/ReluRelu0sequential_1/vgg16/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:         		А╛
5sequential_1/vgg16/block5_conv2/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg16_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ж
&sequential_1/vgg16/block5_conv2/Conv2DConv2D2sequential_1/vgg16/block5_conv1/Relu:activations:0=sequential_1/vgg16/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
│
6sequential_1/vgg16/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg16_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0▐
'sequential_1/vgg16/block5_conv2/BiasAddBiasAdd/sequential_1/vgg16/block5_conv2/Conv2D:output:0>sequential_1/vgg16/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		АЩ
$sequential_1/vgg16/block5_conv2/ReluRelu0sequential_1/vgg16/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:         		А╛
5sequential_1/vgg16/block5_conv3/Conv2D/ReadVariableOpReadVariableOp>sequential_1_vgg16_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ж
&sequential_1/vgg16/block5_conv3/Conv2DConv2D2sequential_1/vgg16/block5_conv2/Relu:activations:0=sequential_1/vgg16/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
│
6sequential_1/vgg16/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_vgg16_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0▐
'sequential_1/vgg16/block5_conv3/BiasAddBiasAdd/sequential_1/vgg16/block5_conv3/Conv2D:output:0>sequential_1/vgg16/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		АЩ
$sequential_1/vgg16/block5_conv3/ReluRelu0sequential_1/vgg16/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:         		А╙
&sequential_1/vgg16/block5_pool/MaxPoolMaxPool2sequential_1/vgg16/block5_conv3/Relu:activations:0*0
_output_shapes
:         А*
ksize
*
paddingVALID*
strides
m
sequential_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"        ┤
sequential_1/flatten_1/ReshapeReshape/sequential_1/vgg16/block5_pool/MaxPool:output:0%sequential_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:         А@а
*sequential_1/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype0╡
sequential_1/dense_2/MatMulMatMul'sequential_1/flatten_1/Reshape:output:02sequential_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЭ
+sequential_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╢
sequential_1/dense_2/BiasAddBiasAdd%sequential_1/dense_2/MatMul:product:03sequential_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А{
sequential_1/dense_2/ReluRelu%sequential_1/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:         АЯ
*sequential_1/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_3_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0┤
sequential_1/dense_3/MatMulMatMul'sequential_1/dense_2/Relu:activations:02sequential_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ь
+sequential_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╡
sequential_1/dense_3/BiasAddBiasAdd%sequential_1/dense_3/MatMul:product:03sequential_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         А
sequential_1/dense_3/SigmoidSigmoid%sequential_1/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:         o
IdentityIdentity sequential_1/dense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:         ╣
NoOpNoOp,^sequential_1/dense_2/BiasAdd/ReadVariableOp+^sequential_1/dense_2/MatMul/ReadVariableOp,^sequential_1/dense_3/BiasAdd/ReadVariableOp+^sequential_1/dense_3/MatMul/ReadVariableOp7^sequential_1/vgg16/block1_conv1/BiasAdd/ReadVariableOp6^sequential_1/vgg16/block1_conv1/Conv2D/ReadVariableOp7^sequential_1/vgg16/block1_conv2/BiasAdd/ReadVariableOp6^sequential_1/vgg16/block1_conv2/Conv2D/ReadVariableOp7^sequential_1/vgg16/block2_conv1/BiasAdd/ReadVariableOp6^sequential_1/vgg16/block2_conv1/Conv2D/ReadVariableOp7^sequential_1/vgg16/block2_conv2/BiasAdd/ReadVariableOp6^sequential_1/vgg16/block2_conv2/Conv2D/ReadVariableOp7^sequential_1/vgg16/block3_conv1/BiasAdd/ReadVariableOp6^sequential_1/vgg16/block3_conv1/Conv2D/ReadVariableOp7^sequential_1/vgg16/block3_conv2/BiasAdd/ReadVariableOp6^sequential_1/vgg16/block3_conv2/Conv2D/ReadVariableOp7^sequential_1/vgg16/block3_conv3/BiasAdd/ReadVariableOp6^sequential_1/vgg16/block3_conv3/Conv2D/ReadVariableOp7^sequential_1/vgg16/block4_conv1/BiasAdd/ReadVariableOp6^sequential_1/vgg16/block4_conv1/Conv2D/ReadVariableOp7^sequential_1/vgg16/block4_conv2/BiasAdd/ReadVariableOp6^sequential_1/vgg16/block4_conv2/Conv2D/ReadVariableOp7^sequential_1/vgg16/block4_conv3/BiasAdd/ReadVariableOp6^sequential_1/vgg16/block4_conv3/Conv2D/ReadVariableOp7^sequential_1/vgg16/block5_conv1/BiasAdd/ReadVariableOp6^sequential_1/vgg16/block5_conv1/Conv2D/ReadVariableOp7^sequential_1/vgg16/block5_conv2/BiasAdd/ReadVariableOp6^sequential_1/vgg16/block5_conv2/Conv2D/ReadVariableOp7^sequential_1/vgg16/block5_conv3/BiasAdd/ReadVariableOp6^sequential_1/vgg16/block5_conv3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:         ЦЦ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+sequential_1/dense_2/BiasAdd/ReadVariableOp+sequential_1/dense_2/BiasAdd/ReadVariableOp2X
*sequential_1/dense_2/MatMul/ReadVariableOp*sequential_1/dense_2/MatMul/ReadVariableOp2Z
+sequential_1/dense_3/BiasAdd/ReadVariableOp+sequential_1/dense_3/BiasAdd/ReadVariableOp2X
*sequential_1/dense_3/MatMul/ReadVariableOp*sequential_1/dense_3/MatMul/ReadVariableOp2p
6sequential_1/vgg16/block1_conv1/BiasAdd/ReadVariableOp6sequential_1/vgg16/block1_conv1/BiasAdd/ReadVariableOp2n
5sequential_1/vgg16/block1_conv1/Conv2D/ReadVariableOp5sequential_1/vgg16/block1_conv1/Conv2D/ReadVariableOp2p
6sequential_1/vgg16/block1_conv2/BiasAdd/ReadVariableOp6sequential_1/vgg16/block1_conv2/BiasAdd/ReadVariableOp2n
5sequential_1/vgg16/block1_conv2/Conv2D/ReadVariableOp5sequential_1/vgg16/block1_conv2/Conv2D/ReadVariableOp2p
6sequential_1/vgg16/block2_conv1/BiasAdd/ReadVariableOp6sequential_1/vgg16/block2_conv1/BiasAdd/ReadVariableOp2n
5sequential_1/vgg16/block2_conv1/Conv2D/ReadVariableOp5sequential_1/vgg16/block2_conv1/Conv2D/ReadVariableOp2p
6sequential_1/vgg16/block2_conv2/BiasAdd/ReadVariableOp6sequential_1/vgg16/block2_conv2/BiasAdd/ReadVariableOp2n
5sequential_1/vgg16/block2_conv2/Conv2D/ReadVariableOp5sequential_1/vgg16/block2_conv2/Conv2D/ReadVariableOp2p
6sequential_1/vgg16/block3_conv1/BiasAdd/ReadVariableOp6sequential_1/vgg16/block3_conv1/BiasAdd/ReadVariableOp2n
5sequential_1/vgg16/block3_conv1/Conv2D/ReadVariableOp5sequential_1/vgg16/block3_conv1/Conv2D/ReadVariableOp2p
6sequential_1/vgg16/block3_conv2/BiasAdd/ReadVariableOp6sequential_1/vgg16/block3_conv2/BiasAdd/ReadVariableOp2n
5sequential_1/vgg16/block3_conv2/Conv2D/ReadVariableOp5sequential_1/vgg16/block3_conv2/Conv2D/ReadVariableOp2p
6sequential_1/vgg16/block3_conv3/BiasAdd/ReadVariableOp6sequential_1/vgg16/block3_conv3/BiasAdd/ReadVariableOp2n
5sequential_1/vgg16/block3_conv3/Conv2D/ReadVariableOp5sequential_1/vgg16/block3_conv3/Conv2D/ReadVariableOp2p
6sequential_1/vgg16/block4_conv1/BiasAdd/ReadVariableOp6sequential_1/vgg16/block4_conv1/BiasAdd/ReadVariableOp2n
5sequential_1/vgg16/block4_conv1/Conv2D/ReadVariableOp5sequential_1/vgg16/block4_conv1/Conv2D/ReadVariableOp2p
6sequential_1/vgg16/block4_conv2/BiasAdd/ReadVariableOp6sequential_1/vgg16/block4_conv2/BiasAdd/ReadVariableOp2n
5sequential_1/vgg16/block4_conv2/Conv2D/ReadVariableOp5sequential_1/vgg16/block4_conv2/Conv2D/ReadVariableOp2p
6sequential_1/vgg16/block4_conv3/BiasAdd/ReadVariableOp6sequential_1/vgg16/block4_conv3/BiasAdd/ReadVariableOp2n
5sequential_1/vgg16/block4_conv3/Conv2D/ReadVariableOp5sequential_1/vgg16/block4_conv3/Conv2D/ReadVariableOp2p
6sequential_1/vgg16/block5_conv1/BiasAdd/ReadVariableOp6sequential_1/vgg16/block5_conv1/BiasAdd/ReadVariableOp2n
5sequential_1/vgg16/block5_conv1/Conv2D/ReadVariableOp5sequential_1/vgg16/block5_conv1/Conv2D/ReadVariableOp2p
6sequential_1/vgg16/block5_conv2/BiasAdd/ReadVariableOp6sequential_1/vgg16/block5_conv2/BiasAdd/ReadVariableOp2n
5sequential_1/vgg16/block5_conv2/Conv2D/ReadVariableOp5sequential_1/vgg16/block5_conv2/Conv2D/ReadVariableOp2p
6sequential_1/vgg16/block5_conv3/BiasAdd/ReadVariableOp6sequential_1/vgg16/block5_conv3/BiasAdd/ReadVariableOp2n
5sequential_1/vgg16/block5_conv3/Conv2D/ReadVariableOp5sequential_1/vgg16/block5_conv3/Conv2D/ReadVariableOp:^ Z
1
_output_shapes
:         ЦЦ
%
_user_specified_namevgg16_input
·
д
,__inference_block4_conv2_layer_call_fn_19924

inputs#
unknown:АА
	unknown_0:	А
identityИвStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block4_conv2_layer_call_and_return_conditional_losses_17679x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         А
 
_user_specified_nameinputs
·
д
,__inference_block3_conv3_layer_call_fn_19874

inputs#
unknown:АА
	unknown_0:	А
identityИвStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         %%А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block3_conv3_layer_call_and_return_conditional_losses_17644x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         %%А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         %%А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         %%А
 
_user_specified_nameinputs
е

Ў
B__inference_dense_2_layer_call_and_return_conditional_losses_19705

inputs2
matmul_readvariableop_resource:
А@А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А@А*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А@
 
_user_specified_nameinputs
О
b
F__inference_block2_pool_layer_call_and_return_conditional_losses_19825

inputs
identityв
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Т
А
G__inference_block1_conv2_layer_call_and_return_conditional_losses_17557

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ы
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЦЦ@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЦЦ@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:         ЦЦ@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:         ЦЦ@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ЦЦ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ЦЦ@
 
_user_specified_nameinputs
·
д
,__inference_block5_conv3_layer_call_fn_20014

inputs#
unknown:АА
	unknown_0:	А
identityИвStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_block5_conv3_layer_call_and_return_conditional_losses_17748x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         		А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         		А: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
ь┤
Н
!__inference__traced_restore_20350
file_prefix3
assignvariableop_dense_2_kernel:
А@А.
assignvariableop_1_dense_2_bias:	А4
!assignvariableop_2_dense_3_kernel:	А-
assignvariableop_3_dense_3_bias:@
&assignvariableop_4_block1_conv1_kernel:@2
$assignvariableop_5_block1_conv1_bias:@@
&assignvariableop_6_block1_conv2_kernel:@@2
$assignvariableop_7_block1_conv2_bias:@A
&assignvariableop_8_block2_conv1_kernel:@А3
$assignvariableop_9_block2_conv1_bias:	АC
'assignvariableop_10_block2_conv2_kernel:АА4
%assignvariableop_11_block2_conv2_bias:	АC
'assignvariableop_12_block3_conv1_kernel:АА4
%assignvariableop_13_block3_conv1_bias:	АC
'assignvariableop_14_block3_conv2_kernel:АА4
%assignvariableop_15_block3_conv2_bias:	АC
'assignvariableop_16_block3_conv3_kernel:АА4
%assignvariableop_17_block3_conv3_bias:	АC
'assignvariableop_18_block4_conv1_kernel:АА4
%assignvariableop_19_block4_conv1_bias:	АC
'assignvariableop_20_block4_conv2_kernel:АА4
%assignvariableop_21_block4_conv2_bias:	АC
'assignvariableop_22_block4_conv3_kernel:АА4
%assignvariableop_23_block4_conv3_bias:	АC
'assignvariableop_24_block5_conv1_kernel:АА4
%assignvariableop_25_block5_conv1_bias:	АC
'assignvariableop_26_block5_conv2_kernel:АА4
%assignvariableop_27_block5_conv2_bias:	АC
'assignvariableop_28_block5_conv3_kernel:АА4
%assignvariableop_29_block5_conv3_bias:	А'
assignvariableop_30_adam_iter:	 )
assignvariableop_31_adam_beta_1: )
assignvariableop_32_adam_beta_2: (
assignvariableop_33_adam_decay: 0
&assignvariableop_34_adam_learning_rate: %
assignvariableop_35_total_1: %
assignvariableop_36_count_1: #
assignvariableop_37_total: #
assignvariableop_38_count: =
)assignvariableop_39_adam_dense_2_kernel_m:
А@А6
'assignvariableop_40_adam_dense_2_bias_m:	А<
)assignvariableop_41_adam_dense_3_kernel_m:	А5
'assignvariableop_42_adam_dense_3_bias_m:=
)assignvariableop_43_adam_dense_2_kernel_v:
А@А6
'assignvariableop_44_adam_dense_2_bias_v:	А<
)assignvariableop_45_adam_dense_3_kernel_v:	А5
'assignvariableop_46_adam_dense_3_bias_v:
identity_48ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9▐
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*Д
value·Bў0B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH╨
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B С
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*╓
_output_shapes├
└::::::::::::::::::::::::::::::::::::::::::::::::*>
dtypes4
220	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOpAssignVariableOpassignvariableop_dense_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_3_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_3_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_4AssignVariableOp&assignvariableop_4_block1_conv1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_5AssignVariableOp$assignvariableop_5_block1_conv1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_6AssignVariableOp&assignvariableop_6_block1_conv2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_7AssignVariableOp$assignvariableop_7_block1_conv2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_8AssignVariableOp&assignvariableop_8_block2_conv1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_9AssignVariableOp$assignvariableop_9_block2_conv1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_10AssignVariableOp'assignvariableop_10_block2_conv2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_11AssignVariableOp%assignvariableop_11_block2_conv2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_12AssignVariableOp'assignvariableop_12_block3_conv1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_13AssignVariableOp%assignvariableop_13_block3_conv1_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_14AssignVariableOp'assignvariableop_14_block3_conv2_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_15AssignVariableOp%assignvariableop_15_block3_conv2_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_16AssignVariableOp'assignvariableop_16_block3_conv3_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_17AssignVariableOp%assignvariableop_17_block3_conv3_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_18AssignVariableOp'assignvariableop_18_block4_conv1_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_19AssignVariableOp%assignvariableop_19_block4_conv1_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_20AssignVariableOp'assignvariableop_20_block4_conv2_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_21AssignVariableOp%assignvariableop_21_block4_conv2_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_22AssignVariableOp'assignvariableop_22_block4_conv3_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_23AssignVariableOp%assignvariableop_23_block4_conv3_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_24AssignVariableOp'assignvariableop_24_block5_conv1_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_25AssignVariableOp%assignvariableop_25_block5_conv1_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_26AssignVariableOp'assignvariableop_26_block5_conv2_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_27AssignVariableOp%assignvariableop_27_block5_conv2_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_28AssignVariableOp'assignvariableop_28_block5_conv3_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_29AssignVariableOp%assignvariableop_29_block5_conv3_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0	*
_output_shapes
:О
AssignVariableOp_30AssignVariableOpassignvariableop_30_adam_iterIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_31AssignVariableOpassignvariableop_31_adam_beta_1Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_32AssignVariableOpassignvariableop_32_adam_beta_2Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_33AssignVariableOpassignvariableop_33_adam_decayIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_34AssignVariableOp&assignvariableop_34_adam_learning_rateIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_35AssignVariableOpassignvariableop_35_total_1Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_36AssignVariableOpassignvariableop_36_count_1Identity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_37AssignVariableOpassignvariableop_37_totalIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_38AssignVariableOpassignvariableop_38_countIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_39AssignVariableOp)assignvariableop_39_adam_dense_2_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_40AssignVariableOp'assignvariableop_40_adam_dense_2_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_dense_3_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adam_dense_3_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_dense_2_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_dense_2_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_45AssignVariableOp)assignvariableop_45_adam_dense_3_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_46AssignVariableOp'assignvariableop_46_adam_dense_3_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ┘
Identity_47Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_48IdentityIdentity_47:output:0^NoOp_1*
T0*
_output_shapes
: ╞
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_48Identity_48:output:0*s
_input_shapesb
`: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Т
Г
G__inference_block5_conv2_layer_call_and_return_conditional_losses_17731

inputs:
conv2d_readvariableop_resource:АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:АА*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		АY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         		Аj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         		Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         		А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         		А
 
_user_specified_nameinputs
▒
G
+__inference_block1_pool_layer_call_fn_19770

inputs
identity╫
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_block1_pool_layer_call_and_return_conditional_losses_17471Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs"╡	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╝
serving_defaultи
M
vgg16_input>
serving_default_vgg16_input:0         ЦЦ;
dense_30
StatefulPartitionedCall:0         tensorflow/serving/predict:Мє
ш
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
№
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
layer_with_weights-5
layer-8
layer_with_weights-6
layer-9
layer-10
layer_with_weights-7
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer-14
layer_with_weights-10
layer-15
layer_with_weights-11
layer-16
layer_with_weights-12
layer-17
 layer-18
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses"
_tf_keras_network
е
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses"
_tf_keras_layer
╗
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

3kernel
4bias"
_tf_keras_layer
╗
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

;kernel
<bias"
_tf_keras_layer
Ж
=0
>1
?2
@3
A4
B5
C6
D7
E8
F9
G10
H11
I12
J13
K14
L15
M16
N17
O18
P19
Q20
R21
S22
T23
U24
V25
326
427
;28
<29"
trackable_list_wrapper
<
30
41
;2
<3"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
х
\trace_0
]trace_1
^trace_2
_trace_32·
,__inference_sequential_1_layer_call_fn_18500
,__inference_sequential_1_layer_call_fn_19063
,__inference_sequential_1_layer_call_fn_19128
,__inference_sequential_1_layer_call_fn_18789┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z\trace_0z]trace_1z^trace_2z_trace_3
╤
`trace_0
atrace_1
btrace_2
ctrace_32ц
G__inference_sequential_1_layer_call_and_return_conditional_losses_19244
G__inference_sequential_1_layer_call_and_return_conditional_losses_19360
G__inference_sequential_1_layer_call_and_return_conditional_losses_18857
G__inference_sequential_1_layer_call_and_return_conditional_losses_18925┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z`trace_0zatrace_1zbtrace_2zctrace_3
╧B╠
 __inference__wrapped_model_17462vgg16_input"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
г
diter

ebeta_1

fbeta_2
	gdecay
hlearning_rate3mО4mП;mР<mС3vТ4vУ;vФ<vХ"
	optimizer
,
iserving_default"
signature_map
"
_tf_keras_input_layer
▌
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses

=kernel
>bias
 p_jit_compiled_convolution_op"
_tf_keras_layer
▌
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses

?kernel
@bias
 w_jit_compiled_convolution_op"
_tf_keras_layer
е
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses"
_tf_keras_layer
т
~	variables
trainable_variables
Аregularization_losses
Б	keras_api
В__call__
+Г&call_and_return_all_conditional_losses

Akernel
Bbias
!Д_jit_compiled_convolution_op"
_tf_keras_layer
ф
Е	variables
Жtrainable_variables
Зregularization_losses
И	keras_api
Й__call__
+К&call_and_return_all_conditional_losses

Ckernel
Dbias
!Л_jit_compiled_convolution_op"
_tf_keras_layer
л
М	variables
Нtrainable_variables
Оregularization_losses
П	keras_api
Р__call__
+С&call_and_return_all_conditional_losses"
_tf_keras_layer
ф
Т	variables
Уtrainable_variables
Фregularization_losses
Х	keras_api
Ц__call__
+Ч&call_and_return_all_conditional_losses

Ekernel
Fbias
!Ш_jit_compiled_convolution_op"
_tf_keras_layer
ф
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Ь	keras_api
Э__call__
+Ю&call_and_return_all_conditional_losses

Gkernel
Hbias
!Я_jit_compiled_convolution_op"
_tf_keras_layer
ф
а	variables
бtrainable_variables
вregularization_losses
г	keras_api
д__call__
+е&call_and_return_all_conditional_losses

Ikernel
Jbias
!ж_jit_compiled_convolution_op"
_tf_keras_layer
л
з	variables
иtrainable_variables
йregularization_losses
к	keras_api
л__call__
+м&call_and_return_all_conditional_losses"
_tf_keras_layer
ф
н	variables
оtrainable_variables
пregularization_losses
░	keras_api
▒__call__
+▓&call_and_return_all_conditional_losses

Kkernel
Lbias
!│_jit_compiled_convolution_op"
_tf_keras_layer
ф
┤	variables
╡trainable_variables
╢regularization_losses
╖	keras_api
╕__call__
+╣&call_and_return_all_conditional_losses

Mkernel
Nbias
!║_jit_compiled_convolution_op"
_tf_keras_layer
ф
╗	variables
╝trainable_variables
╜regularization_losses
╛	keras_api
┐__call__
+└&call_and_return_all_conditional_losses

Okernel
Pbias
!┴_jit_compiled_convolution_op"
_tf_keras_layer
л
┬	variables
├trainable_variables
─regularization_losses
┼	keras_api
╞__call__
+╟&call_and_return_all_conditional_losses"
_tf_keras_layer
ф
╚	variables
╔trainable_variables
╩regularization_losses
╦	keras_api
╠__call__
+═&call_and_return_all_conditional_losses

Qkernel
Rbias
!╬_jit_compiled_convolution_op"
_tf_keras_layer
ф
╧	variables
╨trainable_variables
╤regularization_losses
╥	keras_api
╙__call__
+╘&call_and_return_all_conditional_losses

Skernel
Tbias
!╒_jit_compiled_convolution_op"
_tf_keras_layer
ф
╓	variables
╫trainable_variables
╪regularization_losses
┘	keras_api
┌__call__
+█&call_and_return_all_conditional_losses

Ukernel
Vbias
!▄_jit_compiled_convolution_op"
_tf_keras_layer
л
▌	variables
▐trainable_variables
▀regularization_losses
р	keras_api
с__call__
+т&call_and_return_all_conditional_losses"
_tf_keras_layer
ц
=0
>1
?2
@3
A4
B5
C6
D7
E8
F9
G10
H11
I12
J13
K14
L15
M16
N17
O18
P19
Q20
R21
S22
T23
U24
V25"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
уnon_trainable_variables
фlayers
хmetrics
 цlayer_regularization_losses
чlayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
╤
шtrace_0
щtrace_1
ъtrace_2
ыtrace_32▐
%__inference_vgg16_layer_call_fn_17811
%__inference_vgg16_layer_call_fn_19417
%__inference_vgg16_layer_call_fn_19474
%__inference_vgg16_layer_call_fn_18186┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zшtrace_0zщtrace_1zъtrace_2zыtrace_3
╜
ьtrace_0
эtrace_1
юtrace_2
яtrace_32╩
@__inference_vgg16_layer_call_and_return_conditional_losses_19574
@__inference_vgg16_layer_call_and_return_conditional_losses_19674
@__inference_vgg16_layer_call_and_return_conditional_losses_18260
@__inference_vgg16_layer_call_and_return_conditional_losses_18334┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zьtrace_0zэtrace_1zюtrace_2zяtrace_3
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Ёnon_trainable_variables
ёlayers
Єmetrics
 єlayer_regularization_losses
Їlayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
я
їtrace_02╨
)__inference_flatten_1_layer_call_fn_19679в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zїtrace_0
К
Ўtrace_02ы
D__inference_flatten_1_layer_call_and_return_conditional_losses_19685в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЎtrace_0
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
ўnon_trainable_variables
°layers
∙metrics
 ·layer_regularization_losses
√layer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
э
№trace_02╬
'__inference_dense_2_layer_call_fn_19694в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z№trace_0
И
¤trace_02щ
B__inference_dense_2_layer_call_and_return_conditional_losses_19705в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z¤trace_0
": 
А@А2dense_2/kernel
:А2dense_2/bias
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
■non_trainable_variables
 layers
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
э
Гtrace_02╬
'__inference_dense_3_layer_call_fn_19714в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zГtrace_0
И
Дtrace_02щ
B__inference_dense_3_layer_call_and_return_conditional_losses_19725в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zДtrace_0
!:	А2dense_3/kernel
:2dense_3/bias
-:+@2block1_conv1/kernel
:@2block1_conv1/bias
-:+@@2block1_conv2/kernel
:@2block1_conv2/bias
.:,@А2block2_conv1/kernel
 :А2block2_conv1/bias
/:-АА2block2_conv2/kernel
 :А2block2_conv2/bias
/:-АА2block3_conv1/kernel
 :А2block3_conv1/bias
/:-АА2block3_conv2/kernel
 :А2block3_conv2/bias
/:-АА2block3_conv3/kernel
 :А2block3_conv3/bias
/:-АА2block4_conv1/kernel
 :А2block4_conv1/bias
/:-АА2block4_conv2/kernel
 :А2block4_conv2/bias
/:-АА2block4_conv3/kernel
 :А2block4_conv3/bias
/:-АА2block5_conv1/kernel
 :А2block5_conv1/bias
/:-АА2block5_conv2/kernel
 :А2block5_conv2/bias
/:-АА2block5_conv3/kernel
 :А2block5_conv3/bias
ц
=0
>1
?2
@3
A4
B5
C6
D7
E8
F9
G10
H11
I12
J13
K14
L15
M16
N17
O18
P19
Q20
R21
S22
T23
U24
V25"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
0
Е0
Ж1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ВB 
,__inference_sequential_1_layer_call_fn_18500vgg16_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
¤B·
,__inference_sequential_1_layer_call_fn_19063inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
¤B·
,__inference_sequential_1_layer_call_fn_19128inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ВB 
,__inference_sequential_1_layer_call_fn_18789vgg16_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ШBХ
G__inference_sequential_1_layer_call_and_return_conditional_losses_19244inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ШBХ
G__inference_sequential_1_layer_call_and_return_conditional_losses_19360inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЭBЪ
G__inference_sequential_1_layer_call_and_return_conditional_losses_18857vgg16_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЭBЪ
G__inference_sequential_1_layer_call_and_return_conditional_losses_18925vgg16_input"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
╬B╦
#__inference_signature_wrapper_18998vgg16_input"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
Є
Мtrace_02╙
,__inference_block1_conv1_layer_call_fn_19734в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zМtrace_0
Н
Нtrace_02ю
G__inference_block1_conv1_layer_call_and_return_conditional_losses_19745в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zНtrace_0
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Оnon_trainable_variables
Пlayers
Рmetrics
 Сlayer_regularization_losses
Тlayer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
Є
Уtrace_02╙
,__inference_block1_conv2_layer_call_fn_19754в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zУtrace_0
Н
Фtrace_02ю
G__inference_block1_conv2_layer_call_and_return_conditional_losses_19765в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zФtrace_0
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Хnon_trainable_variables
Цlayers
Чmetrics
 Шlayer_regularization_losses
Щlayer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
ё
Ъtrace_02╥
+__inference_block1_pool_layer_call_fn_19770в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЪtrace_0
М
Ыtrace_02э
F__inference_block1_pool_layer_call_and_return_conditional_losses_19775в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЫtrace_0
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╢
Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
~	variables
trainable_variables
Аregularization_losses
В__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
Є
бtrace_02╙
,__inference_block2_conv1_layer_call_fn_19784в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zбtrace_0
Н
вtrace_02ю
G__inference_block2_conv1_layer_call_and_return_conditional_losses_19795в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zвtrace_0
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
гnon_trainable_variables
дlayers
еmetrics
 жlayer_regularization_losses
зlayer_metrics
Е	variables
Жtrainable_variables
Зregularization_losses
Й__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
Є
иtrace_02╙
,__inference_block2_conv2_layer_call_fn_19804в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zиtrace_0
Н
йtrace_02ю
G__inference_block2_conv2_layer_call_and_return_conditional_losses_19815в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zйtrace_0
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
кnon_trainable_variables
лlayers
мmetrics
 нlayer_regularization_losses
оlayer_metrics
М	variables
Нtrainable_variables
Оregularization_losses
Р__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
ё
пtrace_02╥
+__inference_block2_pool_layer_call_fn_19820в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zпtrace_0
М
░trace_02э
F__inference_block2_pool_layer_call_and_return_conditional_losses_19825в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z░trace_0
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
▒non_trainable_variables
▓layers
│metrics
 ┤layer_regularization_losses
╡layer_metrics
Т	variables
Уtrainable_variables
Фregularization_losses
Ц__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
Є
╢trace_02╙
,__inference_block3_conv1_layer_call_fn_19834в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╢trace_0
Н
╖trace_02ю
G__inference_block3_conv1_layer_call_and_return_conditional_losses_19845в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╖trace_0
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╕non_trainable_variables
╣layers
║metrics
 ╗layer_regularization_losses
╝layer_metrics
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Э__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
Є
╜trace_02╙
,__inference_block3_conv2_layer_call_fn_19854в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╜trace_0
Н
╛trace_02ю
G__inference_block3_conv2_layer_call_and_return_conditional_losses_19865в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╛trace_0
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
┐non_trainable_variables
└layers
┴metrics
 ┬layer_regularization_losses
├layer_metrics
а	variables
бtrainable_variables
вregularization_losses
д__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
Є
─trace_02╙
,__inference_block3_conv3_layer_call_fn_19874в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z─trace_0
Н
┼trace_02ю
G__inference_block3_conv3_layer_call_and_return_conditional_losses_19885в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┼trace_0
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╞non_trainable_variables
╟layers
╚metrics
 ╔layer_regularization_losses
╩layer_metrics
з	variables
иtrainable_variables
йregularization_losses
л__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses"
_generic_user_object
ё
╦trace_02╥
+__inference_block3_pool_layer_call_fn_19890в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╦trace_0
М
╠trace_02э
F__inference_block3_pool_layer_call_and_return_conditional_losses_19895в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╠trace_0
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
═non_trainable_variables
╬layers
╧metrics
 ╨layer_regularization_losses
╤layer_metrics
н	variables
оtrainable_variables
пregularization_losses
▒__call__
+▓&call_and_return_all_conditional_losses
'▓"call_and_return_conditional_losses"
_generic_user_object
Є
╥trace_02╙
,__inference_block4_conv1_layer_call_fn_19904в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╥trace_0
Н
╙trace_02ю
G__inference_block4_conv1_layer_call_and_return_conditional_losses_19915в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╙trace_0
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╘non_trainable_variables
╒layers
╓metrics
 ╫layer_regularization_losses
╪layer_metrics
┤	variables
╡trainable_variables
╢regularization_losses
╕__call__
+╣&call_and_return_all_conditional_losses
'╣"call_and_return_conditional_losses"
_generic_user_object
Є
┘trace_02╙
,__inference_block4_conv2_layer_call_fn_19924в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┘trace_0
Н
┌trace_02ю
G__inference_block4_conv2_layer_call_and_return_conditional_losses_19935в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┌trace_0
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
█non_trainable_variables
▄layers
▌metrics
 ▐layer_regularization_losses
▀layer_metrics
╗	variables
╝trainable_variables
╜regularization_losses
┐__call__
+└&call_and_return_all_conditional_losses
'└"call_and_return_conditional_losses"
_generic_user_object
Є
рtrace_02╙
,__inference_block4_conv3_layer_call_fn_19944в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zрtrace_0
Н
сtrace_02ю
G__inference_block4_conv3_layer_call_and_return_conditional_losses_19955в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zсtrace_0
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
тnon_trainable_variables
уlayers
фmetrics
 хlayer_regularization_losses
цlayer_metrics
┬	variables
├trainable_variables
─regularization_losses
╞__call__
+╟&call_and_return_all_conditional_losses
'╟"call_and_return_conditional_losses"
_generic_user_object
ё
чtrace_02╥
+__inference_block4_pool_layer_call_fn_19960в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zчtrace_0
М
шtrace_02э
F__inference_block4_pool_layer_call_and_return_conditional_losses_19965в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zшtrace_0
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
щnon_trainable_variables
ъlayers
ыmetrics
 ьlayer_regularization_losses
эlayer_metrics
╚	variables
╔trainable_variables
╩regularization_losses
╠__call__
+═&call_and_return_all_conditional_losses
'═"call_and_return_conditional_losses"
_generic_user_object
Є
юtrace_02╙
,__inference_block5_conv1_layer_call_fn_19974в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zюtrace_0
Н
яtrace_02ю
G__inference_block5_conv1_layer_call_and_return_conditional_losses_19985в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zяtrace_0
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Ёnon_trainable_variables
ёlayers
Єmetrics
 єlayer_regularization_losses
Їlayer_metrics
╧	variables
╨trainable_variables
╤regularization_losses
╙__call__
+╘&call_and_return_all_conditional_losses
'╘"call_and_return_conditional_losses"
_generic_user_object
Є
їtrace_02╙
,__inference_block5_conv2_layer_call_fn_19994в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zїtrace_0
Н
Ўtrace_02ю
G__inference_block5_conv2_layer_call_and_return_conditional_losses_20005в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЎtrace_0
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
ўnon_trainable_variables
°layers
∙metrics
 ·layer_regularization_losses
√layer_metrics
╓	variables
╫trainable_variables
╪regularization_losses
┌__call__
+█&call_and_return_all_conditional_losses
'█"call_and_return_conditional_losses"
_generic_user_object
Є
№trace_02╙
,__inference_block5_conv3_layer_call_fn_20014в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z№trace_0
Н
¤trace_02ю
G__inference_block5_conv3_layer_call_and_return_conditional_losses_20025в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z¤trace_0
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
■non_trainable_variables
 layers
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
▌	variables
▐trainable_variables
▀regularization_losses
с__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
ё
Гtrace_02╥
+__inference_block5_pool_layer_call_fn_20030в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zГtrace_0
М
Дtrace_02э
F__inference_block5_pool_layer_call_and_return_conditional_losses_20035в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zДtrace_0
ц
=0
>1
?2
@3
A4
B5
C6
D7
E8
F9
G10
H11
I12
J13
K14
L15
M16
N17
O18
P19
Q20
R21
S22
T23
U24
V25"
trackable_list_wrapper
о
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
 18"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ўBЇ
%__inference_vgg16_layer_call_fn_17811input_2"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЎBє
%__inference_vgg16_layer_call_fn_19417inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЎBє
%__inference_vgg16_layer_call_fn_19474inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ўBЇ
%__inference_vgg16_layer_call_fn_18186input_2"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
СBО
@__inference_vgg16_layer_call_and_return_conditional_losses_19574inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
СBО
@__inference_vgg16_layer_call_and_return_conditional_losses_19674inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ТBП
@__inference_vgg16_layer_call_and_return_conditional_losses_18260input_2"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ТBП
@__inference_vgg16_layer_call_and_return_conditional_losses_18334input_2"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▌B┌
)__inference_flatten_1_layer_call_fn_19679inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
°Bї
D__inference_flatten_1_layer_call_and_return_conditional_losses_19685inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
█B╪
'__inference_dense_2_layer_call_fn_19694inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЎBє
B__inference_dense_2_layer_call_and_return_conditional_losses_19705inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
█B╪
'__inference_dense_3_layer_call_fn_19714inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЎBє
B__inference_dense_3_layer_call_and_return_conditional_losses_19725inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
R
Е	variables
Ж	keras_api

Зtotal

Иcount"
_tf_keras_metric
c
Й	variables
К	keras_api

Лtotal

Мcount
Н
_fn_kwargs"
_tf_keras_metric
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
рB▌
,__inference_block1_conv1_layer_call_fn_19734inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
G__inference_block1_conv1_layer_call_and_return_conditional_losses_19745inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
рB▌
,__inference_block1_conv2_layer_call_fn_19754inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
G__inference_block1_conv2_layer_call_and_return_conditional_losses_19765inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▀B▄
+__inference_block1_pool_layer_call_fn_19770inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
F__inference_block1_pool_layer_call_and_return_conditional_losses_19775inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
рB▌
,__inference_block2_conv1_layer_call_fn_19784inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
G__inference_block2_conv1_layer_call_and_return_conditional_losses_19795inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
рB▌
,__inference_block2_conv2_layer_call_fn_19804inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
G__inference_block2_conv2_layer_call_and_return_conditional_losses_19815inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▀B▄
+__inference_block2_pool_layer_call_fn_19820inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
F__inference_block2_pool_layer_call_and_return_conditional_losses_19825inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
рB▌
,__inference_block3_conv1_layer_call_fn_19834inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
G__inference_block3_conv1_layer_call_and_return_conditional_losses_19845inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
рB▌
,__inference_block3_conv2_layer_call_fn_19854inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
G__inference_block3_conv2_layer_call_and_return_conditional_losses_19865inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
рB▌
,__inference_block3_conv3_layer_call_fn_19874inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
G__inference_block3_conv3_layer_call_and_return_conditional_losses_19885inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▀B▄
+__inference_block3_pool_layer_call_fn_19890inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
F__inference_block3_pool_layer_call_and_return_conditional_losses_19895inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
рB▌
,__inference_block4_conv1_layer_call_fn_19904inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
G__inference_block4_conv1_layer_call_and_return_conditional_losses_19915inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
рB▌
,__inference_block4_conv2_layer_call_fn_19924inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
G__inference_block4_conv2_layer_call_and_return_conditional_losses_19935inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
рB▌
,__inference_block4_conv3_layer_call_fn_19944inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
G__inference_block4_conv3_layer_call_and_return_conditional_losses_19955inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▀B▄
+__inference_block4_pool_layer_call_fn_19960inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
F__inference_block4_pool_layer_call_and_return_conditional_losses_19965inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
рB▌
,__inference_block5_conv1_layer_call_fn_19974inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
G__inference_block5_conv1_layer_call_and_return_conditional_losses_19985inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
рB▌
,__inference_block5_conv2_layer_call_fn_19994inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
G__inference_block5_conv2_layer_call_and_return_conditional_losses_20005inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
рB▌
,__inference_block5_conv3_layer_call_fn_20014inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
G__inference_block5_conv3_layer_call_and_return_conditional_losses_20025inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▀B▄
+__inference_block5_pool_layer_call_fn_20030inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
F__inference_block5_pool_layer_call_and_return_conditional_losses_20035inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
0
З0
И1"
trackable_list_wrapper
.
Е	variables"
_generic_user_object
:  (2total
:  (2count
0
Л0
М1"
trackable_list_wrapper
.
Й	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
':%
А@А2Adam/dense_2/kernel/m
 :А2Adam/dense_2/bias/m
&:$	А2Adam/dense_3/kernel/m
:2Adam/dense_3/bias/m
':%
А@А2Adam/dense_2/kernel/v
 :А2Adam/dense_2/bias/v
&:$	А2Adam/dense_3/kernel/v
:2Adam/dense_3/bias/v╕
 __inference__wrapped_model_17462У=>?@ABCDEFGHIJKLMNOPQRSTUV34;<>в;
4в1
/К,
vgg16_input         ЦЦ
к "1к.
,
dense_3!К
dense_3         ╗
G__inference_block1_conv1_layer_call_and_return_conditional_losses_19745p=>9в6
/в,
*К'
inputs         ЦЦ
к "/в,
%К"
0         ЦЦ@
Ъ У
,__inference_block1_conv1_layer_call_fn_19734c=>9в6
/в,
*К'
inputs         ЦЦ
к ""К         ЦЦ@╗
G__inference_block1_conv2_layer_call_and_return_conditional_losses_19765p?@9в6
/в,
*К'
inputs         ЦЦ@
к "/в,
%К"
0         ЦЦ@
Ъ У
,__inference_block1_conv2_layer_call_fn_19754c?@9в6
/в,
*К'
inputs         ЦЦ@
к ""К         ЦЦ@щ
F__inference_block1_pool_layer_call_and_return_conditional_losses_19775ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ┴
+__inference_block1_pool_layer_call_fn_19770СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ╕
G__inference_block2_conv1_layer_call_and_return_conditional_losses_19795mAB7в4
-в*
(К%
inputs         KK@
к ".в+
$К!
0         KKА
Ъ Р
,__inference_block2_conv1_layer_call_fn_19784`AB7в4
-в*
(К%
inputs         KK@
к "!К         KKА╣
G__inference_block2_conv2_layer_call_and_return_conditional_losses_19815nCD8в5
.в+
)К&
inputs         KKА
к ".в+
$К!
0         KKА
Ъ С
,__inference_block2_conv2_layer_call_fn_19804aCD8в5
.в+
)К&
inputs         KKА
к "!К         KKАщ
F__inference_block2_pool_layer_call_and_return_conditional_losses_19825ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ┴
+__inference_block2_pool_layer_call_fn_19820СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ╣
G__inference_block3_conv1_layer_call_and_return_conditional_losses_19845nEF8в5
.в+
)К&
inputs         %%А
к ".в+
$К!
0         %%А
Ъ С
,__inference_block3_conv1_layer_call_fn_19834aEF8в5
.в+
)К&
inputs         %%А
к "!К         %%А╣
G__inference_block3_conv2_layer_call_and_return_conditional_losses_19865nGH8в5
.в+
)К&
inputs         %%А
к ".в+
$К!
0         %%А
Ъ С
,__inference_block3_conv2_layer_call_fn_19854aGH8в5
.в+
)К&
inputs         %%А
к "!К         %%А╣
G__inference_block3_conv3_layer_call_and_return_conditional_losses_19885nIJ8в5
.в+
)К&
inputs         %%А
к ".в+
$К!
0         %%А
Ъ С
,__inference_block3_conv3_layer_call_fn_19874aIJ8в5
.в+
)К&
inputs         %%А
к "!К         %%Ащ
F__inference_block3_pool_layer_call_and_return_conditional_losses_19895ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ┴
+__inference_block3_pool_layer_call_fn_19890СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ╣
G__inference_block4_conv1_layer_call_and_return_conditional_losses_19915nKL8в5
.в+
)К&
inputs         А
к ".в+
$К!
0         А
Ъ С
,__inference_block4_conv1_layer_call_fn_19904aKL8в5
.в+
)К&
inputs         А
к "!К         А╣
G__inference_block4_conv2_layer_call_and_return_conditional_losses_19935nMN8в5
.в+
)К&
inputs         А
к ".в+
$К!
0         А
Ъ С
,__inference_block4_conv2_layer_call_fn_19924aMN8в5
.в+
)К&
inputs         А
к "!К         А╣
G__inference_block4_conv3_layer_call_and_return_conditional_losses_19955nOP8в5
.в+
)К&
inputs         А
к ".в+
$К!
0         А
Ъ С
,__inference_block4_conv3_layer_call_fn_19944aOP8в5
.в+
)К&
inputs         А
к "!К         Ащ
F__inference_block4_pool_layer_call_and_return_conditional_losses_19965ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ┴
+__inference_block4_pool_layer_call_fn_19960СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ╣
G__inference_block5_conv1_layer_call_and_return_conditional_losses_19985nQR8в5
.в+
)К&
inputs         		А
к ".в+
$К!
0         		А
Ъ С
,__inference_block5_conv1_layer_call_fn_19974aQR8в5
.в+
)К&
inputs         		А
к "!К         		А╣
G__inference_block5_conv2_layer_call_and_return_conditional_losses_20005nST8в5
.в+
)К&
inputs         		А
к ".в+
$К!
0         		А
Ъ С
,__inference_block5_conv2_layer_call_fn_19994aST8в5
.в+
)К&
inputs         		А
к "!К         		А╣
G__inference_block5_conv3_layer_call_and_return_conditional_losses_20025nUV8в5
.в+
)К&
inputs         		А
к ".в+
$К!
0         		А
Ъ С
,__inference_block5_conv3_layer_call_fn_20014aUV8в5
.в+
)К&
inputs         		А
к "!К         		Ащ
F__inference_block5_pool_layer_call_and_return_conditional_losses_20035ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ┴
+__inference_block5_pool_layer_call_fn_20030СRвO
HвE
CК@
inputs4                                    
к ";К84                                    д
B__inference_dense_2_layer_call_and_return_conditional_losses_19705^340в-
&в#
!К
inputs         А@
к "&в#
К
0         А
Ъ |
'__inference_dense_2_layer_call_fn_19694Q340в-
&в#
!К
inputs         А@
к "К         Аг
B__inference_dense_3_layer_call_and_return_conditional_losses_19725];<0в-
&в#
!К
inputs         А
к "%в"
К
0         
Ъ {
'__inference_dense_3_layer_call_fn_19714P;<0в-
&в#
!К
inputs         А
к "К         к
D__inference_flatten_1_layer_call_and_return_conditional_losses_19685b8в5
.в+
)К&
inputs         А
к "&в#
К
0         А@
Ъ В
)__inference_flatten_1_layer_call_fn_19679U8в5
.в+
)К&
inputs         А
к "К         А@█
G__inference_sequential_1_layer_call_and_return_conditional_losses_18857П=>?@ABCDEFGHIJKLMNOPQRSTUV34;<FвC
<в9
/К,
vgg16_input         ЦЦ
p 

 
к "%в"
К
0         
Ъ █
G__inference_sequential_1_layer_call_and_return_conditional_losses_18925П=>?@ABCDEFGHIJKLMNOPQRSTUV34;<FвC
<в9
/К,
vgg16_input         ЦЦ
p

 
к "%в"
К
0         
Ъ ╓
G__inference_sequential_1_layer_call_and_return_conditional_losses_19244К=>?@ABCDEFGHIJKLMNOPQRSTUV34;<Aв>
7в4
*К'
inputs         ЦЦ
p 

 
к "%в"
К
0         
Ъ ╓
G__inference_sequential_1_layer_call_and_return_conditional_losses_19360К=>?@ABCDEFGHIJKLMNOPQRSTUV34;<Aв>
7в4
*К'
inputs         ЦЦ
p

 
к "%в"
К
0         
Ъ │
,__inference_sequential_1_layer_call_fn_18500В=>?@ABCDEFGHIJKLMNOPQRSTUV34;<FвC
<в9
/К,
vgg16_input         ЦЦ
p 

 
к "К         │
,__inference_sequential_1_layer_call_fn_18789В=>?@ABCDEFGHIJKLMNOPQRSTUV34;<FвC
<в9
/К,
vgg16_input         ЦЦ
p

 
к "К         н
,__inference_sequential_1_layer_call_fn_19063}=>?@ABCDEFGHIJKLMNOPQRSTUV34;<Aв>
7в4
*К'
inputs         ЦЦ
p 

 
к "К         н
,__inference_sequential_1_layer_call_fn_19128}=>?@ABCDEFGHIJKLMNOPQRSTUV34;<Aв>
7в4
*К'
inputs         ЦЦ
p

 
к "К         ╩
#__inference_signature_wrapper_18998в=>?@ABCDEFGHIJKLMNOPQRSTUV34;<MвJ
в 
Cк@
>
vgg16_input/К,
vgg16_input         ЦЦ"1к.
,
dense_3!К
dense_3         ╒
@__inference_vgg16_layer_call_and_return_conditional_losses_18260Р=>?@ABCDEFGHIJKLMNOPQRSTUVBв?
8в5
+К(
input_2         ЦЦ
p 

 
к ".в+
$К!
0         А
Ъ ╒
@__inference_vgg16_layer_call_and_return_conditional_losses_18334Р=>?@ABCDEFGHIJKLMNOPQRSTUVBв?
8в5
+К(
input_2         ЦЦ
p

 
к ".в+
$К!
0         А
Ъ ╘
@__inference_vgg16_layer_call_and_return_conditional_losses_19574П=>?@ABCDEFGHIJKLMNOPQRSTUVAв>
7в4
*К'
inputs         ЦЦ
p 

 
к ".в+
$К!
0         А
Ъ ╘
@__inference_vgg16_layer_call_and_return_conditional_losses_19674П=>?@ABCDEFGHIJKLMNOPQRSTUVAв>
7в4
*К'
inputs         ЦЦ
p

 
к ".в+
$К!
0         А
Ъ н
%__inference_vgg16_layer_call_fn_17811Г=>?@ABCDEFGHIJKLMNOPQRSTUVBв?
8в5
+К(
input_2         ЦЦ
p 

 
к "!К         Ан
%__inference_vgg16_layer_call_fn_18186Г=>?@ABCDEFGHIJKLMNOPQRSTUVBв?
8в5
+К(
input_2         ЦЦ
p

 
к "!К         Ам
%__inference_vgg16_layer_call_fn_19417В=>?@ABCDEFGHIJKLMNOPQRSTUVAв>
7в4
*К'
inputs         ЦЦ
p 

 
к "!К         Ам
%__inference_vgg16_layer_call_fn_19474В=>?@ABCDEFGHIJKLMNOPQRSTUVAв>
7в4
*К'
inputs         ЦЦ
p

 
к "!К         А