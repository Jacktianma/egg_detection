/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f1xx_hal.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "motor.h"
#include "stdio.h"
#include "string.h"
#include "egg_ctrl.h"
/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */
extern ADC_HandleTypeDef hadc1;
extern DMA_HandleTypeDef hdma_adc1;
extern TIM_HandleTypeDef htim2;
extern TIM_HandleTypeDef htim3;
extern TIM_HandleTypeDef htim4;
/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */
#define UART_RX_BUFFER_SIZE 256
extern uint8_t uart1_dma_rx_buffer[UART_RX_BUFFER_SIZE];
extern uint8_t uart1_frame_buffer[UART_RX_BUFFER_SIZE];
extern uint16_t uart1_last_index;
extern uint8_t Start;
/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* USER CODE BEGIN EFP */

/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
#define OPT101_Pin GPIO_PIN_1
#define OPT101_GPIO_Port GPIOA
#define EGG_GET_Pin GPIO_PIN_2
#define EGG_GET_GPIO_Port GPIOA
#define EGG_OUT_Pin GPIO_PIN_3
#define EGG_OUT_GPIO_Port GPIOA
#define DIRZ_Pin GPIO_PIN_4
#define DIRZ_GPIO_Port GPIOA
#define DIRC_Pin GPIO_PIN_5
#define DIRC_GPIO_Port GPIOA
#define DIRY1_Pin GPIO_PIN_4
#define DIRY1_GPIO_Port GPIOC
#define DIRY2_Pin GPIO_PIN_5
#define DIRY2_GPIO_Port GPIOC
#define DIRX_Pin GPIO_PIN_2
#define DIRX_GPIO_Port GPIOB
#define Light620_Pin GPIO_PIN_7
#define Light620_GPIO_Port GPIOE
#define Light700_Pin GPIO_PIN_8
#define Light700_GPIO_Port GPIOE
#define Light800_Pin GPIO_PIN_9
#define Light800_GPIO_Port GPIOE
#define Pos_Pin GPIO_PIN_12
#define Pos_GPIO_Port GPIOE
#define MZ_Pin GPIO_PIN_6
#define MZ_GPIO_Port GPIOC
#define MC_Pin GPIO_PIN_7
#define MC_GPIO_Port GPIOC
#define MX_Pin GPIO_PIN_8
#define MX_GPIO_Port GPIOC
#define MY_Pin GPIO_PIN_9
#define MY_GPIO_Port GPIOC

/* USER CODE BEGIN Private defines */

/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
