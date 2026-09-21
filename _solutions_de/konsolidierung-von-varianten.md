---
title: Konsolidierung von Varianten
description: Regelmäßiges Zusammenführen der Varianten, die mehrere
  Kunden teilen, zurück in das Standardprodukt, und Abschalten derer,
  von denen nichts mehr abhängt.
category:
- Architecture
- Business
- Process
problems:
- excessive-customization
- high-maintenance-costs
- code-duplication
- testing-complexity
- long-release-cycles
- increased-cost-of-development
- high-technical-debt
- slow-feature-development
- maintenance-cost-increase
- feature-creep
- technology-stack-fragmentation
- entity-attribute-value-overuse
- core-modification-of-standard-software
- custom-report-sprawl
- reimplemented-standard-functionality
- upgrade-blocked-by-customization
layout: solution
lang: de
en_slug: variant-consolidation
related_solutions:
- slug: customization-cost-attribution
  similarity: 0.8
- slug: explicit-extension-points
  similarity: 0.7
- slug: fit-to-standard-principle
  similarity: 0.7
- slug: feature-usage-measurement
  similarity: 0.65
- slug: consistent-terminology
  similarity: 0.65
- slug: large-scale-refactoring
  similarity: 0.6
---

## Description

Konsolidierung von Varianten ist die wiederkehrende Praxis, die angesammelten kundenspezifischen Varianten zu untersuchen, die, die mehrere Kunden effektiv teilen, zurück in das Standardprodukt zusammenzuführen, und die auszumustern, von denen nichts mehr abhängt. Es ist die Reduktionshälfte der Verwaltung eines anpassbaren Produkts; Erweiterungspunkte verhindern, dass sich neue Variation in den Kern ausbreitet, aber sie tun nichts gegen das, was sich bereits angesammelt hat. Konsolidierung zählt, weil Anpassungsportfolios eine charakteristische Form haben: ein Satz von Varianten, die distinkt aussehen, weil sie separat angefragt und nach ihren Anfragenden benannt wurden, die aber im Wesentlichen dasselbe tun, plus eine erhebliche Anzahl, die tot ist — der anfragende Kunde ging, der Prozess änderte sich, oder das Bedürfnis war vorübergehend. Keine der beiden Gruppen ist sichtbar, ohne dass jemand bewusst hinschaut, und nichts im normalen Betrieb veranlasst je jemanden, hinzuschauen.

## How to Apply ◆

> Varianten sind nach wem benannt, der sie angefragt hat, was genau das ist, was verbirgt, dass drei von ihnen dieselbe Sache sind.

- **Überprüfen Sie in fester Taktung**, jährlich oder halbjährlich. Konsolidierung geschieht nie opportunistisch, weil sie in keinem einzelnen Moment das Dringendste ist, was verfügbar ist.
- **Gruppieren Sie nach dem, was die Variante tut**, nicht nach wer sie angefragt hat. Jede Variante in einem Satz Verhaltensbeschreibung zu beschreiben, mit entferntem Kundennamen, reicht üblicherweise aus, um zu offenbaren, dass mehrere dasselbe sind. Dieser Schritt allein halbiert häufig die scheinbare Anzahl.
- **Identifizieren Sie die toten** mithilfe von Evidenz: Ist der anfragende Kunde noch Kunde, wird der Codepfad noch ausgeführt, wurde die Variante seit Jahren angefasst. Tote Varianten sind die günstigste Reduktion und erfordern keine Verhandlung mit irgendjemandem.
- **Verallgemeinern Sie die geteilten in das Standardprodukt**, wo mehrere Kunden im Wesentlichen dasselbe Verhalten wollen. Die Verallgemeinerung ist üblicherweise eine Konfigurationsoption statt einer neuen Variante, und sie beseitigt mehrere Wartungslasten auf einmal.
- **Verhandeln Sie die knappen Fehlschläge.** Wo drei Varianten sich nur geringfügig unterscheiden, ist der Unterschied häufig verhandelbar — Kunden akzeptierten eine Variante, weil sie angeboten wurde, nicht weil das spezifische Detail zählte. Zu fragen ist günstig und funktioniert oft.
- **Muster Sie bewusst aus, mit Ankündigung und einem Migrationspfad.** Eine ohne Warnung entfernte Variante ist ein Support-Vorfall und ein Vertrauensproblem, und der resultierende Ruf macht die nächste Konsolidierung schwerer.
- **Speisen Sie die Kostenzahlen in das Gespräch ein.** Ein Kunde, der gebeten wird, zum Standardverhalten zu wechseln, reagiert anders, wenn die Anfrage begleitet wird von dem, was ihre Variante kostet und was sie im Gegenzug bekommen — üblicherweise schnellere Upgrades und schnelleren Zugang zu neuen Fähigkeiten.
- **Akzeptieren Sie die, die bleiben müssen.** Manche Varianten sind genuin durch Regulierung, Vertrag oder den echten Prozess eines Kunden erforderlich. Sie als dauerhaft zu benennen, mit einem Grund, schließt das Review, statt sie als ewige Kandidaten zu belassen.
- **Protokollieren Sie die Reduktion.** Ausgemusterte Varianten, verallgemeinerte Varianten und die geschätzte freigesetzte Wartung. Konsolidierung produziert kein sichtbares Feature, sodass ihre Ergebnisse bewusst berichtet werden müssen, sonst wird das nächste Review nicht finanziert.

## Tradeoffs ⇄

> Konsolidierung ist das Einzige, was ein angesammeltes Variantenportfolio reduziert, aber sie erfordert Kundenverhandlung und produziert nichts, worum Kunden gebeten haben.

**Vorteile:**

- Die Anzahl der Varianten sinkt absolut, was die Testmatrixgröße, Upgrade-Kosten und die Last für jede zukünftige Änderung reduziert.
- Die Verallgemeinerung geteilten Verhaltens in das Produkt verwandelt mehrere Wartungsverbindlichkeiten in ein einziges unterstütztes Feature, das alle Kunden bekommen.
- Tote Varianten werden im Wesentlichen kostenlos gefunden und entfernt, abgesehen vom Hinsehen, und sie sind üblicherweise ein erheblicher Anteil des Portfolios.
- Kunden, die auf Standardverhalten umgezogen sind, erhalten Upgrades und neue Fähigkeiten schneller, was ein echter Vorteil ist und die Verhandlung möglich macht.
- Regelmäßiges Review verhindert, dass das Portfolio monoton wächst, was es tut, wenn es niemand untersucht.

**Kosten und Risiken:**

- Es erfordert Kundengespräche, gegen die sich das Account-Management sträuben könnte, besonders wo die Beziehung brüchig ist.
- Die Ausmusterung einer Variante, die sich als tragend für den Prozess eines Kunden herausstellt, ist ein ernster Vorfall und schädigt Vertrauen breit, nicht nur mit diesem Kunden.
- Verallgemeinerung kann eine Konfigurationsfläche produzieren, die selbst komplex ist — mehrere Varianten, ersetzt durch ein Feature mit acht Optionen, ist nicht offensichtlich eine Verbesserung.
- Die Arbeit liefert nichts, worum irgendein Kunde gebeten hat, was es schwierig macht, sie gegen Anfragen zu priorisieren, auf die jemand wartet.
- Verhandlung über knappe Fehlschläge braucht Zeit pro Kunde und schlägt häufig fehl, sodass der Aufwand aufgewendet wird, ob die Konsolidierung geschieht oder nicht.

## How It Could Be

Ein Anbieter überprüfte zum ersten Mal seit sechs Jahren 47 Kundenvarianten und beschrieb jede in einem Satz mit entferntem Kundennamen. Elf stellten sich als drei Gruppen heraus, die dasselbe taten: vier Wege, eine Bestätigungs-E-Mail zu unterdrücken, vier Wege, einen Referenzcode zu einer Rechnung hinzuzufügen, und drei Wege, eine Summe zu runden. Jede Gruppe wurde über ungefähr drei Wochen in ein einziges konfigurierbares Feature verallgemeinert. Neun weitere Varianten waren tot — sechs anfragende Kunden waren gegangen, zwei bezogen sich auf einen Prozess, den der Kunde seitdem aufgegeben hatte, und einer war laut den Protokollen nie in Produktion ausgeführt worden. Diese wurden mit einer Benachrichtigung entfernt, ohne Einwand. Das Portfolio fiel innerhalb eines Quartals von 47 auf 30, und die Release-Validierungsmatrix schrumpfte entsprechend.

Die Verhandlung über knappe Fehlschläge war gemischter und aufschlussreicher. Sechs Varianten unterschieden sich nur in einem Datumsformat auf einem gedruckten Dokument. Der Anbieter näherte sich allen sechs Kunden mit dem Angebot des Standardformats. Vier akzeptierten sofort, einer akzeptierte im Austausch für eine andere kleine Änderung, die sie sich gewünscht hatten, und einer lehnte ab mit der Begründung, dass ihr nachgelagertes System das Dokument parste — was sich als wahr und wichtig herausstellte, und diese Variante wurde als dauerhaft protokolliert, mit dem angehängten Grund. Das Account-Team des Anbieters, anfänglich dagegen, Kunden bezüglich des Entfernens von Dingen anzusprechen, berichtete anschließend, dass fünf von sechs Gesprächen positiv gewesen waren, weil das Angebot als schnellere Upgrades gerahmt worden war, statt als Rückzug.
