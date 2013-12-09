#import <cassert>

#import "ObjectViewController.h"
#import "AxisTitleInspector.h"
#import "AxisFontInspector.h"

//C++ (ROOT) imports.
#import "TObject.h"
#import "TAxis.h"

//It's mm file == C++, consts have internal linkage.
const float minTitleOffset = 0.1f;
const float maxTitleOffset = 10.f;
const float titleOffsetStep = 0.01f;

const float minTitleSize = 0.01f;
const float maxTitleSize = 1.f;
const float titleSizeStep = 0.01f;

const float totalHeight = 400.f;
const float tabBarHeight = 49.f;

@implementation AxisTitleInspector {
   __weak IBOutlet UITextField *titleField;
   __weak IBOutlet UISwitch *centered;
   __weak IBOutlet UISwitch *rotated;
   __weak IBOutlet UILabel *offsetLabel;
   __weak IBOutlet UILabel *sizeLabel;
   __weak IBOutlet UIButton *plusSizeBtn;
   __weak IBOutlet UIButton *minusSizeBtn;

   __weak ObjectViewController *controller;
   
   TAxis *object;
   float offset;
   float titleSize;

}

//____________________________________________________________________________________________________
+ (CGRect) inspectorFrame
{
   return CGRectMake(0.f, tabBarHeight, 250.f, totalHeight - tabBarHeight);
}

//____________________________________________________________________________________________________
- (id) initWithNibName : (NSString *) nibNameOrNil bundle : (NSBundle *) nibBundleOrNil
{

   self = [super initWithNibName : nibNameOrNil bundle : nibBundleOrNil];

   if (self)
      [self view];

   return self;
}

//____________________________________________________________________________________________________
- (void) didReceiveMemoryWarning
{
    // Releases the view if it doesn't have a superview.
    [super didReceiveMemoryWarning];
    
    // Release any cached data, images, etc that aren't in use.
}

#pragma mark - View lifecycle

//____________________________________________________________________________________________________
- (void) viewDidLoad
{
    [super viewDidLoad];
    // Do any additional setup after loading the view from its nib.
}

//____________________________________________________________________________________________________
- (BOOL) shouldAutorotateToInterfaceOrientation : (UIInterfaceOrientation) interfaceOrientation
{
#pragma unused(interfaceOrientation)

	return YES;
}

//____________________________________________________________________________________________________
- (void) setObjectController : (ObjectViewController *) c
{
   assert(c != nil && "setObjectController:, parameter 'c' is nil");
   controller = c;
}

//____________________________________________________________________________________________________
- (void) setObject : (TObject *) o
{
   assert(o != nullptr && "setObject:, parameter 'o' is null");

   object = dynamic_cast<TAxis *>(o);
   //I do not test the result of cast, it's done one level up.

   const char *axisTitle = object->GetTitle();
   if (!axisTitle || !*axisTitle)
      titleField.text = @"";
   else
      titleField.text = [NSString stringWithFormat : @"%s", axisTitle];
      
   centered.on = object->GetCenterTitle();
   rotated.on = object->GetRotateTitle();
   
   offset = object->GetTitleOffset();
   offsetLabel.text = [NSString stringWithFormat:@"%.2f", offset];
   
   titleSize = object->GetTitleSize();
   if (titleSize > maxTitleSize || titleSize < minTitleSize) {//this is baaad
      titleSize = minTitleSize;
      object->SetTitleSize(titleSize);
      [controller objectWasModifiedUpdateSelection : NO];
   }

   sizeLabel.text = [NSString stringWithFormat:@"%.2f", titleSize];
}

//____________________________________________________________________________________________________
- (IBAction) showTitleFontInspector
{
   AxisFontInspector *fontInspector = [[AxisFontInspector alloc] initWithNibName : @"AxisFontInspector" mode : ROOT_IOSObjectInspector::afimTitleFont];

   [fontInspector setObjectController : controller];
   [fontInspector setObject : object];
   
   [self.navigationController pushViewController : fontInspector animated : YES];
}

//____________________________________________________________________________________________________
- (IBAction) textFieldDidEndOnExit : (id) sender
{
   object->SetTitle([titleField.text cStringUsingEncoding : [NSString defaultCStringEncoding]]);
   [controller objectWasModifiedUpdateSelection : NO];
}

//____________________________________________________________________________________________________
- (IBAction) textFieldEditingDidEnd : (id) sender
{
   [sender resignFirstResponder];
}

//____________________________________________________________________________________________________
- (IBAction) centerTitle
{
   object->CenterTitle(centered.on);
   [controller objectWasModifiedUpdateSelection : NO];
}

//____________________________________________________________________________________________________
- (IBAction) rotateTitle
{
   object->RotateTitle(rotated.on);
   [controller objectWasModifiedUpdateSelection : NO];
}

//____________________________________________________________________________________________________
- (IBAction) plusOffset
{
   if (offset + titleOffsetStep > maxTitleOffset)
      return;
   
   offset += titleOffsetStep;
   offsetLabel.text = [NSString stringWithFormat:@"%.2f", offset];
   object->SetTitleOffset(offset);
   
   [controller objectWasModifiedUpdateSelection : NO];
}

//____________________________________________________________________________________________________
- (IBAction) minusOffset
{
   if (offset - titleOffsetStep < minTitleOffset)
      return;
   
   offset -= titleOffsetStep;
   offsetLabel.text = [NSString stringWithFormat:@"%.2f", offset];
   object->SetTitleOffset(offset);
   
   [controller objectWasModifiedUpdateSelection : NO];
}

//____________________________________________________________________________________________________
- (IBAction) plusSize
{
   if (titleSize + titleSizeStep > maxTitleSize)
      return;
   
   titleSize += titleSizeStep;
   sizeLabel.text = [NSString stringWithFormat:@"%.2f", titleSize];
   object->SetTitleSize(titleSize);
   
   [controller objectWasModifiedUpdateSelection : NO];
}

//____________________________________________________________________________________________________
- (IBAction) minusSize
{
   if (titleSize - titleSizeStep < minTitleSize)
      return;
   
   titleSize -= titleSizeStep;
   sizeLabel.text = [NSString stringWithFormat:@"%.2f", titleSize];
   object->SetTitleSize(titleSize);
   
   [controller objectWasModifiedUpdateSelection : NO];
}


@end
