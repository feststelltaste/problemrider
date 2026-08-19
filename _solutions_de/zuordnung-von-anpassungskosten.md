---
title: Zuordnung von Anpassungskosten
description: Nachverfolgung, was die Wartung jeder kundenspezifischen Variante kostet,
  und Zuordnung zu demjenigen, der sie angefragt hat, sodass eine Zusage zu einer
  bepreisten Entscheidung wird.
category:
- Business
- Management
- Process
problems:
- excessive-customization
- eager-to-please-stakeholders
- high-maintenance-costs
- increased-cost-of-development
- market-pressure
- short-term-focus
- feature-creep
- difficulty-quantifying-benefits
- maintenance-cost-increase
- competing-priorities
- product-direction-chaos
- invisible-nature-of-technical-debt
- core-modification-of-standard-software
- custom-report-sprawl
- reimplemented-standard-functionality
layout: solution
lang: de
en_slug: customization-cost-attribution
related_solutions:
- slug: variant-consolidation
  similarity: 0.8
- slug: explicit-extension-points
  similarity: 0.7
- slug: fit-to-standard-principle
  similarity: 0.65
- slug: total-cost-of-ownership-transparency
  similarity: 0.65
- slug: cost-of-delay
  similarity: 0.65
- slug: feature-usage-measurement
  similarity: 0.6
---

## Description

Zuordnung von Anpassungskosten erfasst, was es kostet, jede Variante am Leben zu erhalten — die für die Wartung aufgewendete Entwicklerzeit, das zusätzliche Testing, den auferlegten Upgrade-Aufwand — und ordnet diese Kosten dem Kunden, dem Deal oder der Abteilung zu, die sie angefragt hat. Sie adressiert den strukturellen Grund, warum sich Anpassung anhäuft: Die Person, die einer Variante zustimmt, bezahlt nicht dafür. Ein Vertriebsmitarbeiter, der einen Deal abschließt, oder eine Führungskraft, die einen Großkunden entgegenkommt, verursacht Kosten, die Jahre später auf einem Engineering-Budget landen, das niemand mit dieser Entscheidung verbindet. Unter dieser Anordnung ist Zustimmung immer lokal rational, und die Anhäufung ist garantiert. Zuordnung verbietet Anpassung nicht; sie macht die Entscheidung bepreist, sodass die Organisation entscheiden kann, ob eine Variante ihre Kosten wert ist, statt die Antwort ein Jahrzehnt später zu entdecken.

## How to Apply ◆

> Niemand in einem stark angepassten Produkt kann sagen, was eine einzelne Anpassung kostet, was genau der Grund ist, warum es so viele davon gibt.

- **Identifizieren Sie Varianten als diskrete, benannte Dinge** mit einem Eigentümer und einer anfragenden Partei. Anpassung, die nur als verstreute Bedingungen existiert, kann nicht bepreist werden, sodass die Bestandsaufnahme die Voraussetzung ist und meist bereits von sich aus aufschlussreich ist.
- **Ordnen Sie den direkten Aufwand zu**: Zeit, die für variantenspezifische Fehler aufgewendet wird, Zeit für die Anpassung während Releases und ihr zurechenbarer Support-Aufwand. Selbst grobe Verfolgung über eine kleine Zahl von Varianten erzeugt innerhalb eines Quartals ein nutzbares Bild.
- **Beziehen Sie die Upgrade-Kosten ein**, die meist die größte Komponente und die unsichtbarste sind. Was es braucht, um eine Installation mit dieser Variante auf eine neue Version zu bringen, gemessen an dem, was es letztes Mal tatsächlich brauchte.
- **Beziehen Sie die Steuer auf alles andere ein.** Eine Variante, die immer berücksichtigt werden muss, wenn sich ein gemeinsames Modul ändert, erlegt Arbeit, die nichts mit ihr zu tun hat, Kosten auf. Dies ist diffus und real, und eine grobe Zuweisung ist besser, als es als null zu behandeln.
- **Berichten Sie pro Variante, pro Jahr**, in denselben Begriffen, die das Geschäft für den Umsatz nutzt, den die Variante sichern sollte. Der Vergleich zwischen den beiden ist der ganze Punkt und häufig unangenehm.
- **Bringen Sie die Zahl in die Entscheidung, bevor sie getroffen wird.** Eine Anfrage, die mit einer geschätzten jährlichen Wartungskostenangabe bewertet wird, ist ein anderes Gespräch als eine, die nur nach Implementierungsaufwand bewertet wird. Implementierung ist die Anzahlung; Wartung ist der Kredit.
- **Erwägen Sie, dafür zu berechnen.** Wo das kommerzielle Modell es erlaubt, verwandelt eine explizite laufende Gebühr für eine Variante die Disziplin von einem internen Argument in einen Markttest — und Kunden lehnen häufig ab, sobald die Variante einen Preis hat.
- **Überprüfen Sie das Portfolio jährlich** und identifizieren Sie Varianten, deren Kosten jeden plausiblen Wert übersteigen. Dies sind Kandidaten für die Stilllegung, und das Stilllegungsgespräch ist mit einer angehängten Zahl weit einfacher.
- **Erfassen Sie die Entscheidungen, die trotzdem getroffen werden.** Manche Varianten werden trotz ungünstiger Kosten aus strategischen Gründen genehmigt. Dies als bewusste Entscheidung zu erfassen ist ein legitimes Ergebnis und bewahrt die Glaubwürdigkeit der Praxis.

## Tradeoffs ⇄

> Zuordnung verwandelt unsichtbare, sich anhäufende Kosten in eine bepreiste Entscheidung, aber die Messung ist ungenau und die Befunde erzeugen Konflikt mit den Personen, die die Varianten angefragt haben.

**Vorteile:**

- Die Entscheidung zur Anpassung wird bepreist statt kostenlos, was das Verhalten an dem Punkt ändert, an dem die Anhäufung tatsächlich beginnt.
- Varianten, deren Kosten ihren Wert übersteigen, werden identifizierbar, und ihre Stilllegung ist der einzige Eingriff, der die Last absolut reduziert.
- Vertrieb und Produkt erhalten die Information, die sie brauchen, um eine Anpassung gegen etwas anderes einzutauschen, die sie derzeit nicht haben.
- Die Engineering-Kosten kommerzieller Entscheidungen werden für die Entscheidungsträger sichtbar, was eine normalerweise vollständig fehlende Feedback-Schleife schließt.
- Das Berechnen für Varianten liefert, wo möglich, einen echten Markttest, ob eine Anpassung genug gewollt wird, um dafür zu bezahlen.

**Kosten und Risiken:**

- Zuordnung ist ungenau. Gemeinsame Arbeit ist schwer zuzuweisen, und jede Zahl kann von jedem angefochten werden, dem die Schlussfolgerung nicht gefällt.
- Die Aufwandsverfolgung pro Variante ist administrativer Overhead für Entwickler, und sie verfällt schnell, wenn die resultierenden Zahlen nie genutzt werden.
- Die Befunde erzeugen Konflikt mit Vertrieb und Account-Management, deren Anreize die Praxis direkt konterkariert.
- Die Bepreisung einer von einem strategisch wichtigen Kunden angefragten Variante kann eine Zahl erzeugen, die trotzdem übergangen wird, was riskiert, dass die Praxis als sinnlos wahrgenommen wird.
- Der alleinige Fokus auf Kosten ignoriert, dass manche Varianten strategischen Wert über ihren Umsatz hinaus haben, und eine rein kostengetriebene Portfolioüberprüfung wird empfehlen, Dinge stillzulegen, die sie nicht sollte.

## How It Could Be

Ein Anbieter mit 34 Installationen verfolgte variantenzuordenbaren Aufwand über zwei Quartale gegen eine Liste von 47 benannten Anpassungen. Die Verteilung war extrem: 6 Varianten machten etwa 60 Prozent des zuordenbaren Aufwands aus, und 19 hatten überhaupt keinen messbaren Aufwand verbraucht. Eine Variante — ein maßgeschneiderter Abrechnungsexport für einen vor acht Jahren gewonnenen Kunden — kostete geschätzt 34 Entwicklertage im Jahr, gegen einen jährlichen Vertragswert, den der Account Manager als einen ihrer kleinsten bestätigte. Als einzelner Vergleich präsentiert, wurde die Variante in einem Gespräch mit dem Kunden stillgelegt, der den Standardexport nach einer zweistündigen Durchsprache akzeptierte. Niemand hatte in acht Jahren gefragt, weil niemand die Kosten kannte.

Der zukunftsgerichtete Effekt erwies sich als wichtiger als die Stilllegungen. Der Anbieter fügte jeder Anpassungsanfrage neben der Implementierungsschätzung eine geschätzte jährliche Wartungszahl hinzu. Im folgenden Jahr wurden 14 Anfragen auf diese Weise bewertet. Fünf wurden wie zuvor genehmigt. Vier wurden stattdessen durch Erweiterung des Standardprodukts erfüllt, sobald der Vergleich zeigte, dass eine Variante teurer war als Generalisierung. Drei wurden abgelehnt, und die Kunden akzeptierten die Entscheidung, als ihnen die Begründung gezeigt wurde. Zwei wurden genehmigt, wobei der Kunde eine explizite jährliche Gebühr zahlte. Der Vertriebsleiter, der die Praxis anfangs abgelehnt hatte, beschrieb die Wartungszahl später als das stärkste Verhandlungsinstrument, das er hatte, weil sie eine Anfrage in einen Tausch statt in eine Erwartung verwandelte.
