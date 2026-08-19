"use client"

import {
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react"

import type { PickingInfo } from "@deck.gl/core"
import { GeoJsonLayer } from "@deck.gl/layers"
import { DeckGL } from "@deck.gl/react"
import type {
  Feature,
  FeatureCollection,
  Geometry,
} from "geojson"
import { Map as MapLibre } from "react-map-gl/maplibre"

import type {
  AnalyticsStateSummary,
  BrazilStateCode,
} from "@/server/analytics/analytics-state-summary"
import type {
  BusinessDecision,
  BusinessPriority,
  StateBusinessDecision,
} from "@/server/analytics/business-decision-v1"

type RgbaColor = [
  number,
  number,
  number,
  number,
]

type StateGeometryProperties = {
  state_code: string
  state_name: string
}

type StateMapProperties =
  StateGeometryProperties & {
    orderCount: number
    gmv: number
    aov: number
    lateDeliveryRate: number | null
    averageReviewScore: number | null
    decision: BusinessDecision
    priority: BusinessPriority
  }

type StateFeature = Feature<
  Geometry,
  StateMapProperties
>

type StateSelection = {
  stateCode: BrazilStateCode
  stateName: string
}

type AnalyticsStateMapProps = {
  states: AnalyticsStateSummary[]
  decisions: StateBusinessDecision[]
  selectedStateCode: BrazilStateCode | null
  onStateSelect: (
    selection: StateSelection | null
  ) => void
}

const INITIAL_VIEW_STATE = {
  longitude: -52.5,
  latitude: -14.2,
  zoom: 3.2,
  pitch: 0,
  bearing: 0,
}

const BASEMAP_STYLE =
  "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"

const DECISION_COLORS: Record<
  BusinessDecision,
  RgbaColor
> = {
  RECOVER_SERVICE: [205, 72, 72, 215],
  PROTECT_VALUE: [54, 132, 113, 205],
  EXPAND: [66, 118, 196, 205],
  INVESTIGATE: [218, 151, 53, 210],
  MONITOR: [155, 171, 190, 175],
}

const LEGEND_ITEMS: Array<{
  decision: BusinessDecision
  label: string
}> = [
  {
    decision: "RECOVER_SERVICE",
    label: "Recover Service",
  },
  {
    decision: "PROTECT_VALUE",
    label: "Protect Value",
  },
  {
    decision: "INVESTIGATE",
    label: "Investigate",
  },
  {
    decision: "MONITOR",
    label: "Monitor",
  },
]

export function AnalyticsStateMap({
  states,
  decisions,
  selectedStateCode,
  onStateSelect,
}: AnalyticsStateMapProps) {
  const [geometry, setGeometry] =
    useState<
      FeatureCollection<
        Geometry,
        StateGeometryProperties
      > | null
    >(null)

  const [geometryError, setGeometryError] =
    useState(false)

  useEffect(() => {
    let cancelled = false

    async function loadGeometry() {
      try {
        const response = await fetch(
          "/geo/brazil-states.geojson"
        )

        if (!response.ok) {
          throw new Error(
            "Failed to load Brazil state geometry."
          )
        }

        const data =
          (await response.json()) as FeatureCollection<
            Geometry,
            StateGeometryProperties
          >

        if (
          data.type !== "FeatureCollection" ||
          data.features.length !== 27
        ) {
          throw new Error(
            "Brazil state geometry is incomplete."
          )
        }

        if (!cancelled) {
          setGeometry(data)
        }
      } catch (error) {
        console.error(
          "Brazil state geometry unavailable:",
          error
        )

        if (!cancelled) {
          setGeometryError(true)
        }
      }
    }

    void loadGeometry()

    return () => {
      cancelled = true
    }
  }, [])

  const stateLookup = useMemo(
    () =>
      new Map<
        string,
        AnalyticsStateSummary
      >(
        states.map((state) => [
          state.stateCode,
          state,
        ])
      ),
    [states]
  )

  const decisionLookup = useMemo(
    () =>
      new Map<
        string,
        StateBusinessDecision
      >(
        decisions.map((decision) => [
          decision.stateCode,
          decision,
        ])
      ),
    [decisions]
  )

  const mapData = useMemo(() => {
    if (!geometry) {
      return null
    }

    const features = geometry.features.map(
      (feature): StateFeature => {
        const stateCode =
          feature.properties.state_code

        const state =
          stateLookup.get(stateCode)

        const decision =
          decisionLookup.get(stateCode)

        if (!state) {
          throw new Error(
            `No analytics data for ${stateCode}.`
          )
        }

        if (!decision) {
          throw new Error(
            `No business decision for ${stateCode}.`
          )
        }

        return {
          ...feature,
          properties: {
            ...feature.properties,

            orderCount: state.orderCount,
            gmv: state.gmv,
            aov: state.aov,

            lateDeliveryRate:
              state.lateDeliveryRate,

            averageReviewScore:
              state.averageReviewScore,

            decision: decision.decision,
            priority: decision.priority,
          },
        }
      }
    )

    return {
      type: "FeatureCollection",
      features,
    } as FeatureCollection<
      Geometry,
      StateMapProperties
    >
  }, [
    decisionLookup,
    geometry,
    stateLookup,
  ])

  const layers = useMemo(() => {
    if (!mapData) {
      return []
    }

    return [
      new GeoJsonLayer<StateMapProperties>({
        id: "brazil-state-business-actions",
        data: mapData,

        pickable: true,
        autoHighlight: true,
        highlightColor: [
          15,
          23,
          42,
          35,
        ],

        filled: true,
        stroked: true,

        lineWidthUnits: "pixels",
        lineWidthMinPixels: 1,

        getFillColor: (feature) =>
          getDecisionFillColor(
            feature.properties.decision
          ),

        getLineColor: (feature) =>
          feature.properties.state_code ===
          selectedStateCode
            ? [15, 23, 42, 255]
            : [255, 255, 255, 230],

        getLineWidth: (feature) =>
          feature.properties.state_code ===
          selectedStateCode
            ? 3
            : 1,

        updateTriggers: {
          getLineColor:
            selectedStateCode,
          getLineWidth:
            selectedStateCode,
        },
      }),
    ]
  }, [
    mapData,
    selectedStateCode,
  ])

  const handleClick = useCallback(
    (info: PickingInfo) => {
      const object =
        info.object as StateFeature | null

      if (!object) {
        return
      }

      const state =
        stateLookup.get(
          object.properties.state_code
        )

      if (!state) {
        return
      }

      if (
        selectedStateCode ===
        state.stateCode
      ) {
        onStateSelect(null)
        return
      }

      onStateSelect({
        stateCode: state.stateCode,
        stateName:
          object.properties.state_name,
      })
    },
    [
      onStateSelect,
      selectedStateCode,
      stateLookup,
    ]
  )

  if (geometryError) {
    return (
      <div className="flex h-[520px] items-center justify-center rounded-lg border text-sm text-muted-foreground">
        Brazil state geometry could not be loaded.
      </div>
    )
  }

  if (!mapData) {
    return (
      <div className="flex h-[520px] items-center justify-center rounded-lg border text-sm text-muted-foreground">
        Loading Brazil state map...
      </div>
    )
  }

  return (
    <div className="relative h-[520px] overflow-hidden rounded-lg border">
      <DeckGL
        initialViewState={
          INITIAL_VIEW_STATE
        }
        controller
        layers={layers}
        onClick={handleClick}
        getTooltip={getTooltip}
      >
        <MapLibre
          mapStyle={BASEMAP_STYLE}
        />
      </DeckGL>

      <div className="pointer-events-none absolute left-3 top-3 z-10 rounded-md border bg-background/95 px-3 py-2 shadow-sm backdrop-blur">
        <div className="mb-2 text-xs font-medium">
          Business action
        </div>

        <div className="space-y-1.5">
          {LEGEND_ITEMS.map(
            ({ decision, label }) => {
              const color =
                DECISION_COLORS[decision]

              return (
                <div
                  key={decision}
                  className="flex items-center gap-2 text-xs"
                >
                  <span
                    className="h-2.5 w-2.5 rounded-sm"
                    style={{
                      backgroundColor: `rgb(${color[0]} ${color[1]} ${color[2]})`,
                    }}
                  />

                  <span>{label}</span>
                </div>
              )
            }
          )}
        </div>
      </div>
    </div>
  )
}

function getDecisionFillColor(
  decision: BusinessDecision
): RgbaColor {
  return DECISION_COLORS[decision]
}

function formatDecision(
  decision: BusinessDecision
): string {
  return {
    RECOVER_SERVICE: "Recover Service",
    PROTECT_VALUE: "Protect Value",
    EXPAND: "Expand",
    INVESTIGATE: "Investigate",
    MONITOR: "Monitor",
  }[decision]
}

function formatCurrency(
  value: number
): string {
  return `R$${value.toLocaleString(
    undefined,
    {
      maximumFractionDigits: 0,
    }
  )}`
}

function getTooltip(
  info: PickingInfo
) {
  const object =
    info.object as StateFeature | null

  if (!object) {
    return null
  }

  const state = object.properties

  const lateDelivery =
    state.lateDeliveryRate === null
      ? "No evidence"
      : `${(
          state.lateDeliveryRate * 100
        ).toFixed(1)}%`

  const reviewScore =
    state.averageReviewScore === null
      ? "No evidence"
      : state.averageReviewScore.toFixed(
          2
        )

  return {
    text: [
      `${state.state_name} (${state.state_code})`,
      `${formatDecision(state.decision)} · ${state.priority}`,
      `GMV: ${formatCurrency(state.gmv)}`,
      `Late delivery: ${lateDelivery}`,
      `Review score: ${reviewScore}`,
    ].join("\n"),
  }
}