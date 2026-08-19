---
title: Explizite Erweiterungspunkte
description: Definition einer begrenzten, versionierten Menge von Stellen, an denen
  kundenspezifisches Verhalten andocken darf, sodass Variation an den Rändern statt
  im gesamten Kern lebt.
category:
- Architecture
- Business
- Code
problems:
- excessive-customization
- entity-attribute-value-overuse
- high-technical-debt
- testing-complexity
- slow-feature-development
- increased-cost-of-development
- long-release-cycles
- regression-bugs
- tight-coupling-issues
- high-maintenance-costs
- knowledge-silos
- feature-creep
- eager-to-please-stakeholders
- schema-evolution-paralysis
- core-modification-of-standard-software
- upgrade-blocked-by-customization
- voided-vendor-support
layout: solution
lang: de
en_slug: explicit-extension-points
related_solutions:
- slug: variant-consolidation
  similarity: 0.7
- slug: fit-to-standard-principle
  similarity: 0.7
- slug: customizing
  similarity: 0.7
- slug: customization-cost-attribution
  similarity: 0.7
- slug: customization-under-version-control
  similarity: 0.65
- slug: modularization-and-bounded-contexts
  similarity: 0.65
---

## Description

Explizite Erweiterungspunkte sind eine definierte, begrenzte, versionierte Menge von Stellen, an denen kunden- oder standortspezifisches Verhalten andocken darf — eine Menge von Hooks, eine Regelschnittstelle, eine definierte Konfigurationsfläche — mit der Regel, dass Variation nur dort existieren darf. Die Alternative, die die meisten stark angepassten Systeme tatsächlich haben, ist unbegrenzte Anpassung: Jeder Teil des Kerns kann für jede Installation bedingt geändert werden, sodass Variation überall verteilt ist und der Kern keine Definition hat. Die Unterscheidung bestimmt, ob ein Produkt upgradefähig bleibt. Wenn Variation hinter einer festgelegten Schnittstelle lebt, kann der Kern frei geändert werden, solange die Schnittstelle hält, und jede Installation kann die Änderung übernehmen. Wenn sie durch den Kern gewoben ist, muss jede Änderung gegen jede Variante durchdacht werden, was der Zustand ist, der das Produkt schließlich vollständig an der Weiterentwicklung hindert.

## How to Apply ◆

> Die entscheidende Frage ist nicht, ob Anpassung erlaubt werden soll, sondern wo sie leben darf, und die meisten Legacy-Produkte haben sie nie beantwortet.

- **Definieren Sie zuerst den Kern.** Was das Standardprodukt tut, ohne Vorbehalt. Bis dies als festgelegte Sache existiert, kann keine Anfrage als innerhalb oder außerhalb davon bewertet werden, und jede Anfrage ist daher innerhalb davon.
- **Zählen Sie die Erweiterungspunkte bewusst auf** und halten Sie die Menge klein. Jeder davon ist eine Verpflichtung, die Sie über Versionen hinweg pflegen werden, sodass eine große Fläche eine große dauerhafte Verbindlichkeit ist. Leiten Sie sie aus dem ab, was Kunden tatsächlich gebraucht haben, mittels der bestehenden Anpassungen als Beleg.
- **Versionieren und dokumentieren Sie sie wie eine öffentliche Schnittstelle**, weil das ist, was sie sind. Die Erweiterung eines Kunden beim Upgrade brechen zu lassen ist ein Support-Vorfall, unabhängig davon, ob Sie die Schnittstelle als intern betrachten.
- **Machen Sie den Kern unfähig, spezifische Kunden zu kennen.** Keine Bedingung, geknüpft an einen Installationsidentifikator, irgendwo. Diese Regel ist die operative Form der gesamten Praxis, und sie ist prüfbar — eine Suche nach solchen Bedingungen misst Compliance direkt.
- **Leiten Sie neue Anfragen an den Erweiterungsmechanismus**, und wo eine Anfrage so nicht erfüllt werden kann, behandeln Sie das als Designfrage darüber, ob der Erweiterungsfläche etwas fehlt, statt als Lizenz, den Kern zu modifizieren.
- **Geben Sie Erweiterungen ihre eigenen Tests**, mit ihnen zusammen besessen, sodass die Variation eines Kunden unabhängig verifiziert wird, statt die Testmatrix des Kerns auszudehnen.
- **Migrieren Sie bestehende Anpassungen schrittweise.** Nehmen Sie die in den ohnehin geänderten Bereichen und verschieben Sie sie als Teil dieser Arbeit hinter die Erweiterungsfläche. Eine vollständige Migration wird nicht finanziert; eine opportunistische häuft sich an.
- **Veröffentlichen Sie, was die Erweiterungspunkte nicht abdecken.** Explizit darüber zu sein, was nicht angepasst werden kann, ist so wertvoll wie die Schnittstelle selbst, weil es das ist, was erlaubt, eine Anfrage mit einem Grund abzulehnen statt durch Verhandlung.
- **Überprüfen Sie die Fläche periodisch.** Erweiterungspunkte, die nichts nutzt, sollten entfernt werden; solche, die jeder Kunde identisch nutzt, sind Kandidaten für die Beförderung in den Kern, da eine universell genutzte Erweiterung verkleidete Produktfunktionalität ist.

## Tradeoffs ⇄

> Eine begrenzte Erweiterungsfläche ist das, was ein anpassbares Produkt upgradefähig hält, aber sie gut zu gestalten ist echt schwer, und die Grenze wird konstant auf die Probe gestellt.

**Vorteile:**

- Der Kern kann sich frei weiterentwickeln, weil Änderungen nur den Erweiterungsvertrag statt jeder Installationsvariante respektieren müssen.
- Upgrades werden wieder routinemäßig, was das Muster umkehrt, bei dem Kunden zurückfallen und ihre Installationen weiter auseinanderdriften.
- Die Testmatrix hört auf sich zu vervielfachen, da Erweiterungen gegen die Schnittstelle getestet werden, statt jede Kombination im Kern zu verifizieren.
- Anfragen erhalten eine festgelegte Antwort — innerhalb der Fläche, außerhalb davon oder ein Grund, die Fläche zu erweitern — statt von wem auch immer am stärksten drängt entschieden zu werden.
- Kundenspezifischer Code wird auffindbar und zuordenbar, statt durch den Kern verteilt zu sein, wo ihn niemand finden oder bepreisen kann.

**Kosten und Risiken:**

- Die Gestaltung von Erweiterungspunkten erfordert die Antizipation dessen, was variieren muss, und falsch gestaltete Punkte sind schlimmer als keine — sie schränken ein, ohne zu ermöglichen.
- Die Fläche ist eine dauerhafte Verpflichtung. Sobald Kunden dagegen bauen, ist ihre Änderung ein Breaking Change mit der gesamten damit verbundenen Koordination.
- Manche echten Anfragen werden außerhalb jeder vernünftigen Fläche fallen, und die Disziplin erfordert, gelegentlich Nein zu Umsatz zu sagen.
- Die Migration bestehender Anpassungen ist langsam, und während des Übergangs trägt das System beide Muster.
- Übermäßig allgemeine Erweiterungsmechanismen driften darauf zu, eine Programmierumgebung innerhalb des Produkts zu werden, an welchem Punkt die Anpassungen so undurchsichtig sind wie die Kernmodifikationen, die sie ersetzten.

## How It Could Be

Ein Anbieter von Lagersoftware hatte kundenspezifische Logik an über 60 Stellen über ihren Kern verteilt, geknüpft an einen Standortidentifikator. Das Upgrade eines Standorts brauchte zwei bis vier Wochen Beratung, und elf von 34 Standorten waren mehr als ein Jahr im Rückstand. Sie definierten den Kern zum ersten Mal explizit — eine zweiwöchige Übung, die größtenteils Argumentation war — und leiteten sieben Erweiterungspunkte aus dem ab, was die bestehenden Anpassungen tatsächlich taten: drei Regel-Hooks im Kommissionierfluss, einen Dokumentvorlagenmechanismus, zwei Event-Handler und ein definiertes Konfigurationsschema. Neue Anfragen gingen an diese Fläche. Bestehende Anpassungen wurden opportunistisch migriert, wann immer ihr Bereich angefasst wurde. Nach achtzehn Monaten waren 44 der 60 verschoben, keine Standortbedingung blieb im Kommissionierfluss, und das Upgrade für einen migrierten Standort war auf unter einen Tag gefallen.

Die Regel, dass der Kern keinen Kunden benennen darf, erwies sich als der durchsetzbare Teil. Sie war im Build prüfbar — eine Suche nach Standortidentifikatoren in Kernmodulen — und sie ließ den Build in den ersten zwei Monaten viermal fehlschlagen, jedes Mal einen Entwickler erwischend, der unter Terminplandruck die vertraute Abkürzung nahm. Die Einschätzung des Teams war, dass die Erweiterungspunkte selbst nur die halbe Intervention waren; die andere Hälfte war, dass „eine Bedingung für diesen Kunden hinzufügen" aufgehört hatte, eine verfügbare Option zu sein, was das Designgespräch erzwang, das die Erweiterungsfläche dann beantwortete.
