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

type StateGeometryProperties = {
  state_code: string
  state_name: string
}

type StateMapProperties =
  StateGeometryProperties & {
    orderCount: number
    gmv: number
    aov: number
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

export function AnalyticsStateMap({
  states,
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

        if (!state) {
          throw new Error(
            `No analytics data for ${stateCode}.`
          )
        }

        return {
          ...feature,
          properties: {
            ...feature.properties,
            orderCount: state.orderCount,
            gmv: state.gmv,
            aov: state.aov,
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
  }, [geometry, stateLookup])

  const maxOrders = useMemo(
    () =>
      Math.max(
        ...states.map(
          (state) => state.orderCount
        )
      ),
    [states]
  )

  const layers = useMemo(() => {
    if (!mapData) {
      return []
    }

    return [
      new GeoJsonLayer<StateMapProperties>({
        id: "brazil-state-orders",
        data: mapData,

        pickable: true,
        autoHighlight: true,
        filled: true,
        stroked: true,

        lineWidthUnits: "pixels",
        lineWidthMinPixels: 1,

        getFillColor: (feature) =>
          getOrderFillColor(
            feature.properties.orderCount,
            maxOrders
          ),

        getLineColor: (feature) =>
          feature.properties.state_code ===
          selectedStateCode
            ? [20, 30, 45, 255]
            : [255, 255, 255, 220],

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
    maxOrders,
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
    </div>
  )
}

function getOrderFillColor(
  orders: number,
  maxOrders: number
): [
  number,
  number,
  number,
  number,
] {
  const ratio =
    maxOrders === 0
      ? 0
      : Math.sqrt(
          orders / maxOrders
        )

  return [
    Math.round(220 - 175 * ratio),
    Math.round(235 - 125 * ratio),
    Math.round(250 - 45 * ratio),
    190,
  ]
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

  return {
    text: [
      `${state.state_name} (${state.state_code})`,
      `Orders: ${formatInteger(
        state.orderCount
      )}`,
      `GMV: ${formatBRL(
        state.gmv
      )}`,
      `AOV: ${formatBRL(
        state.aov
      )}`,
    ].join("\n"),
  }
}

function formatInteger(
  value: number
): string {
  return new Intl.NumberFormat(
    "en-US"
  ).format(value)
}

function formatBRL(
  value: number
): string {
  return new Intl.NumberFormat(
    "en-US",
    {
      style: "currency",
      currency: "BRL",
    }
  ).format(value)
}
