/**
 * Zack Hatzis (zdh5)
 * Taylor Stephens (tms279)
 * Adam Fofana (adf64)
 * 
 * This program makes use of a machine learning classifier to detect certain words
 * from audio waveform inputs. As an example of a fun use case, we've included a game
 * that displays the "Stroop Effect". J. R. Stroop conducted experiments in 1935 that
 * indicated that when asked to read out the names of colors written in colored text,
 * it is much harder for people to read out the color of a word than the word itself 
 * when the color of the text does not match the color that the word spells. 
 * 
 * HARDWARE CONNECTIONS
 *  - GPIO 16 ---> VGA Hsync
 *  - GPIO 17 ---> VGA Vsync
 *  - GPIO 18 ---> 330 ohm resistor ---> VGA Red
 *  - GPIO 19 ---> 330 ohm resistor ---> VGA Green
 *  - GPIO 20 ---> 330 ohm resistor ---> VGA Blue
 *  - RP2040 GND ---> VGA GND
 * 
 * MODEL CLASSIFICATIONS
 * 00_red
 * 01_blue
 * 02_green
 * 03_yellow
 * 04_magenta
 * 05_cyan
 * 06_white
*/


#include "edge-impulse-sdk/classifier/ei_run_classifier.h"
//Standard Libraries
#include <stdio.h>
#include <stdlib.h>
// PICO libaries
#include "hardware/gpio.h"
#include "hardware/uart.h"
#include "hardware/dma.h"
#include "hardware/adc.h"
#include "hardware/sync.h"
#include "hardware/pio.h"
#include "hardware/irq.h"

#include "pico/stdio_usb.h"
#include "pico/stdlib.h"

extern "C" {
  // Include protothreads
  #include "pt_cornell_rp2040_v1.h"
  // Include vga libraries
  #include "vga_graphics.h"
}


// === the fixed point macros (16.15) ========================================
typedef signed int fix15 ;
#define multfix15(a,b) ((fix15)((((signed long long)(a))*((signed long long)(b)))>>15))
#define float2fix15(a) ((fix15)((a)*32768.0)) // 2^15
#define fix2float15(a) ((float)(a)/32768.0)
#define absfix15(a) abs(a) 
#define int2fix15(a) ((fix15)(a << 15))
#define fix2int15(a) ((int)(a >> 15))
#define char2fix15(a) (fix15)(((fix15)(a)) << 15)
#define divfix(a,b) (fix15)( (((signed long long)(a)) << 15) / (b))

#define NUM_BITS_PIXEL 24
#define NUM_LEDS 3



const uint LED_PIN = 25;
const uint RED_LED_PIN = 9;
const uint GREEN_LED_PIN = 8;
const uint YELLOW_LED_PIN = 7;
const uint BUTTON_PIN = 10;
//probabiltiy thereshold to detect a result
const float p = 0.6; 

ei_impulse_result_t result = {nullptr};

//DMA Channels for LED Strip
int high_channel = 2;
int value_channel = 3;
int low_channel = 4; 

static const uint32_t resetValue = (1ul << 15);


static uint32_t transmitBuffer[NUM_BITS_PIXEL*NUM_LEDS];

static int r[8];
static int g[8];
static int b[8];

/////////////////////////// ADC configuration ////////////////////////////////
// ADC Channel and pin
#define ADC_CHAN 0
#define ADC_PIN 26
// Number of samples per FFT
#define NUM_SAMPLES 1024
// Number of samples per FFT, minus 1
#define NUM_SAMPLES_M_1 1023
// Length of short (16 bits) minus log2 number of samples (10)
#define SHIFT_AMOUNT 6
// Log2 number of samples
#define LOG2_NUM_SAMPLES 10
// Sample rate (Hz)
#define Fs_ADC 10000.0
// ADC clock rate (unmutable!)
#define ADCCLK 48000000.0

// DMA channels for sampling ADC (VGA driver uses 0 and 1)
int sample_chan = 2 ;
int control_chan = 3 ;

// Max and min macros
#define max(a,b) ((a>b)?a:b)
#define min(a,b) ((a<b)?a:b)

// 0.4 in fixed point (used for alpha max plus beta min)
fix15 zero_point_4 = float2fix15(0.4) ;

// Here's where we'll have the DMA channel put ADC samples
uint8_t sample_array[NUM_SAMPLES] ;
// And here's where we'll copy those samples for FFT calculation
fix15 fr[NUM_SAMPLES] ;
fix15 fi[NUM_SAMPLES] ;

// Sine table for the FFT calculation
fix15 Sinewave[NUM_SAMPLES]; 
// Hann window table for FFT calculation
fix15 window[NUM_SAMPLES]; 

// Pointer to address of start of sample buffer
uint8_t * sample_address_pointer = &sample_array[0] ;

/////////////////////////// VGA Stuff ////////////////////////////////
// semaphore
static struct pt_sem vga_semaphore ;

// character array
char screentext[40];
int display ;
int clear ;

int colors [7] = {RED, GREEN, BLUE, YELLOW, MAGENTA, CYAN, WHITE};

//should match color order
char disp_words [7][8] = {"RED", "GREEN", "BLUE","YELLOW","MAGENTA","CYAN","WHITE"};

//should be sorted by length
char words [8][8] = {"RED","BLUE","CYAN","GREEN","BLACK","WHITE","YELLOW","MAGENTA"};

int streak;


int col;
int wrd;
int disp;


static const float features[] = {};


int raw_feature_get_data(size_t offset, size_t length, float *out_ptr)
{
  memcpy(out_ptr, features + offset, length * sizeof(float));
  return 0;
}

typedef struct {
  int16_t *buffer;
  int16_t *buffer_filtered;
  uint8_t buf_ready;
  uint32_t buf_count;
  uint32_t n_samples;
} inference_t; 

static volatile inference_t inference; 

// amplifier has a bias of VCC/2. Since VCC is 3.3V out, this is 1.65V
#define BIAS ((int16_t) (1.65f * 4095)/3.3f) //Bias of the amplifier
volatile int ix_buffer = 0;
volatile bool is_buffer_ready = false;
volatile int ii = 0;

static void print_raw_audio() {
  for(int i = 0; i < EI_CLASSIFIER_RAW_SAMPLE_COUNT; ++i) {
    printf("%d\n", inference.buffer[i]);
  }
}

bool timer_ISR(struct repeating_timer *t) {
  if (ix_buffer < EI_CLASSIFIER_RAW_SAMPLE_COUNT) {
    if(ii==1000){
      ei_printf("ix_buffer: %d\n", ix_buffer);
      ii = 0;
    }
    //ei_printf("ii: %d\n", ii);
    ++ii;
    int16_t v = (int16_t)(adc_read() - BIAS);
    // int16_t v = (int16_t)(adc_read());
    inference.buffer[ix_buffer] = v;
    ++ix_buffer;
  }
  else {
    is_buffer_ready = true;
  }

  return true; 
}

static bool microphone_inference_start(uint32_t n_samples) {
  inference.buffer = (int16_t *)malloc(n_samples * sizeof(int16_t));

  if(inference.buffer == NULL) {
      return false;
  }

  inference.buf_count  = 0;
  inference.n_samples  = n_samples;
  inference.buf_ready  = 0;

  return true;
}

static bool microphone_inference_record(void) {
  ix_buffer = 0;
  is_buffer_ready = false;
  unsigned int sampling_period = 1000000 / 16000; //62 us
  
  struct repeating_timer timer_0; 


  // add_repeating_timer_us(sampling_period, 
  //       timer_ISR, NULL, &timer_0);

  //for some reason gets mad if you try to do -sampling_period?
  add_repeating_timer_us(-62, 
         timer_ISR, NULL, &timer_0);
  ei_printf("Waiting for buffer\n");
  while(!is_buffer_ready);
  ei_printf("Buffer ready\n");

  cancel_repeating_timer(&timer_0);

  return true ;
}

static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr) {
    numpy::int16_to_float(&inference.buffer[offset], out_ptr, length);

    return 0;
}

void clear_vga(){
  fillRect(0, 0, 640, 480, BLACK);
}

void vga_display(int clear){
  if(clear){
    fillRect(0, 0, 640, 480, BLACK);
  }
  else {
    setTextSize(10) ;
    setTextColor(colors[col]);
    sprintf(screentext, words[wrd]) ;
    //different cursor positions based on length of word
    //word array is sorted manually by length
    if(wrd>6){setCursor(90, 200) ;}
    else if(wrd>5){setCursor(100, 200) ;}
    else if(wrd>2){setCursor(120, 200) ;}
    else if(wrd>0){setCursor(150, 200) ;}
    else{setCursor(200, 200) ;}
    writeString(screentext) ;

  }

}


void correct_display(int correct){
  setTextSize(2) ;
  setTextColor(WHITE);
  if (correct == 2){ //classification failed
    sprintf(screentext, "Classification failed") ;
  }
  else if(correct==1){
    sprintf(screentext, "Nice job! Detected %s",  disp_words[disp]) ; 
  }
  else{
    sprintf(screentext, "Incorrect. Detected %s",  disp_words[disp]) ;
  } 
  // erase the counter
  fillRect(500, 20, 120, 100, BLACK);
 
  setCursor(180, 100);
  writeString(screentext) ;

  //redraw score counter
  setTextSize(2);
  setTextColor(WHITE);
  sprintf(screentext, "Streak: %d", streak);
  setCursor(500, 20);
  writeString(screentext);
  
}



static PT_THREAD (protothread_main(struct pt *pt))
{
  PT_BEGIN(pt) ;
  streak = 0;
    while (true)
  { 
    if (gpio_get(BUTTON_PIN) == 0) {
   
      vga_display(1);
      ei_printf("Button pressed\n");
      // display = (rand() % 2 + 1);
      // ei_printf("display is %d\n", display);
      srand(to_ms_since_boot(get_absolute_time()));
      col = (rand() % 7);
      wrd = (rand() % 8);

      // printf( "random time output: %d\n", to_ms_since_boot(get_absolute_time()));


      vga_display(0);

      //display score counter
      setTextSize(2);
      setTextColor(WHITE);
      sprintf(screentext, "Streak: %d", streak);
      setCursor(500, 20);
      writeString(screentext);

      // PT_YIELD_usec(10) ;
      sleep_ms(700);

      microphone_inference_record();

      // blink LED
      // gpio_put(LED_PIN, !gpio_get(LED_PIN));
      gpio_put(LED_PIN, 0);
      ei_printf("Initializing...\n");

      // the features are stored into flash, and we don't want to load everything into RAM
      signal_t features_signal;
      features_signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
      features_signal.get_data = &microphone_audio_signal_get_data;

      // invoke the impulse
      EI_IMPULSE_ERROR res = run_classifier(&features_signal, &result, false);

      // ei_printf("run_classifier returned: %d\n", res);

      if (res != 0)
        return 1;

      ei_printf("Predictions (DSP: %d ms., Classification: %d ms., Anomaly: %d ms.): \n",
                result.timing.dsp, result.timing.classification, result.timing.anomaly);
      
      gpio_put(RED_LED_PIN, 0);
      gpio_put(GREEN_LED_PIN, 0);
      gpio_put(YELLOW_LED_PIN, 0);
      if(result.classification[0].value > p){ //RED
        // gpio_put(RED_LED_PIN, !gpio_get(RED_LED_PIN));
        if(colors[col] == RED){
          ei_printf("Got red!\n");
          disp = col;
          streak++;
          correct_display(1);
          gpio_put(LED_PIN, 1);
        }
        else{
          disp = 0;
          streak = 0;
          correct_display(0);
          ei_printf("Too bad...\n");
          }
        gpio_put(RED_LED_PIN, 1);
      }
      else if(result.classification[1].value > p){ //GREEN
        if(colors[col] == GREEN){
          ei_printf("Got green!\n");
          disp = col;
          gpio_put(LED_PIN, 1);
          streak++;
          correct_display(1);
        }
        else{ei_printf(
          "Too bad...\n");
          disp = 1;
          streak = 0;
          correct_display(0);
        }
        gpio_put(GREEN_LED_PIN, 1);
      }
      else if(result.classification[2].value > p){ //BLUE
        if(colors[col] == BLUE){ 
          ei_printf("Got blue!\n");
          disp = col;
          gpio_put(LED_PIN, 1);
          streak++;
          correct_display(1);
        }
        else{ 
          ei_printf("Too bad...\n");
          disp = 2;
          streak = 0;
          correct_display(0);
          }
        
      }
      else if(result.classification[3].value > p){ //YELLOW
        if(colors[col] == YELLOW){ 
          disp = col;
          gpio_put(LED_PIN, 1);
          streak++;
          correct_display(1);
        }
        else{ 
          ei_printf("Too bad...\n");
          disp = 3;
          streak = 0;
          correct_display(0);
          }
        ei_printf("Detected yellow!\n");
        gpio_put(YELLOW_LED_PIN, 1);
      }
      else if(result.classification[4].value > p){ //YELLOW
        if(colors[col] == MAGENTA){ 
          disp = col;
          gpio_put(LED_PIN, 1);
          streak++;
          correct_display(1);
        }
        else{ 
          ei_printf("Too bad...\n");
          disp = 4;
          streak = 0;
          correct_display(0);
          }
        ei_printf("Detected magenta!\n");
      }
      else if(result.classification[5].value > p){ //CYAN
        if(colors[col] == CYAN){ 
          disp = col;
          gpio_put(LED_PIN, 1);
          streak++;
          correct_display(1);
        }
        else{ 
          ei_printf("Too bad...\n");
          disp = 5;
          streak = 0;
          correct_display(0);
          }
        ei_printf("Detected cyan!\n");
      }
      else if(result.classification[6].value > p){ //WHITE
        if(colors[col] == WHITE){ 
          disp = col;
          gpio_put(LED_PIN, 1);
          streak++;
          correct_display(1);
        }
        else{ 
          ei_printf("Too bad...\n");
          disp = 6;
          streak = 0;
          correct_display(0);
          }
        ei_printf("Detected white!\n");
      }
      else{
        streak = 0;
        correct_display(2);
      }


      // print the predictions
      ei_printf("[");
      for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++)
      {
        ei_printf("%.5f", result.classification[ix].value);


  #if EI_CLASSIFIER_HAS_ANOMALY == 1
          ei_printf(", ");
  #else
          if (ix != EI_CLASSIFIER_LABEL_COUNT - 1)
          {
            ei_printf(", ");
          }
  #endif
        }
  #if EI_CLASSIFIER_HAS_ANOMALY == 1
        printf("%.3f", result.anomaly);
  #endif
        printf("]\n");

        // PT_YIELD_usec(10) ;
        // vga_display(1);
        ei_sleep(500);
        
    }
  }
  PT_END(pt);
}


int main()
{
  //stdio_usb_init();
  stdio_init_all();

  // initialize VGA
  initVGA() ;

  //maybe replace this with something from button presses
  //srand(47);

  // sprintf(pt_serial_out_buffer, "Initializing 000... \n");
  // serial_write ;
  printf("Initializing... hiiii\n");

  gpio_init(LED_PIN);
  gpio_set_dir(LED_PIN, GPIO_OUT);
  gpio_init(RED_LED_PIN);
  gpio_set_dir(RED_LED_PIN, GPIO_OUT);
  gpio_init(GREEN_LED_PIN);
  gpio_set_dir(GREEN_LED_PIN, GPIO_OUT);
  gpio_init(YELLOW_LED_PIN);
  gpio_set_dir(YELLOW_LED_PIN, GPIO_OUT);

  gpio_init(BUTTON_PIN);
  gpio_set_dir(BUTTON_PIN, GPIO_IN);
  gpio_pull_up(BUTTON_PIN);

  gpio_init(15);
  gpio_set_dir(15, GPIO_OUT);

  
  ///////////////////////////////////////////////////////////////////////////////
  // ================= ADC CONFIGURATION FROM LAB 1 ==========================
  //////////////////////////////////////////////////////////////////////////////
  // Init GPIO for analogue use: hi-Z, no pulls, disable digital input buffer.
  adc_gpio_init(ADC_PIN);

  // Initialize the ADC harware
  // (resets it, enables the clock, spins until the hardware is ready)
  adc_init() ;

  // Select analog mux input (0...3 are GPIO 26, 27, 28, 29; 4 is temp sensor)
  adc_select_input(ADC_CHAN) ;

  // Setup the FIFO
  adc_fifo_setup(
      true,    // Write each completed conversion to the sample FIFO
      true,    // Enable DMA data request (DREQ)
      1,       // DREQ (and IRQ) asserted when at least 1 sample present
      false,   // We won't see the ERR bit because of 8 bit reads; disable.
      true     // Shift each sample to 8 bits when pushing to FIFO
  );

  // Divisor of 0 -> full speed. Free-running capture with the divider is
  // equivalent to pressing the ADC_CS_START_ONCE button once per `div + 1`
  // cycles (div not necessarily an integer). Each conversion takes 96
  // cycles, so in general you want a divider of 0 (hold down the button
  // continuously) or > 95 (take samples less frequently than 96 cycle
  // intervals). This is all timed by the 48 MHz ADC clock. This is setup
  // to grab a sample at 10kHz (48Mhz/10kHz - 1)
  adc_set_clkdiv(ADCCLK/Fs_ADC);


  // Populate the sine table and Hann window table
  int ii;
  for (ii = 0; ii < NUM_SAMPLES; ii++) {
      Sinewave[ii] = float2fix15(sin(6.283 * ((float) ii) / (float)NUM_SAMPLES));
      window[ii] = float2fix15(0.5 * (1.0 - cos(6.283 * ((float) ii) / ((float)NUM_SAMPLES))));
  }

  // ////////////////////////////////////////////////////////////////////////////

  if (microphone_inference_start(EI_CLASSIFIER_RAW_SAMPLE_COUNT) == false) {
    printf("Failed to setup audio sampling \r\n");
  }

  // pt_add_thread(protothread_vga);
  pt_add_thread(protothread_main);
  pt_schedule_start ;
  
}